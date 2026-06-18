import os
import yaml
import torch
import torchaudio
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from funasr.frontends.wav_frontend import WavFrontend
import time
import subprocess
import psutil
import gc
import ctypes

def print_system_status(stage_name):
    # 内存信息
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024 ** 3)
    used_mem = mem.used / (1024 ** 3)
    process_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    
    # 显存信息
    gpu_info_str = "N/A"
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        total_gpu, used_gpu = map(float, res.strip().split(","))
        gpu_percent = (used_gpu / total_gpu) * 100
        
        # 进程显存
        process_gpu_mem = 0.0
        try:
            apps_res = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            current_pid = os.getpid()
            for line in apps_res.strip().split("\n"):
                if line.strip():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        pid = int(parts[0].strip())
                        used_mem_val = float(parts[1].strip())
                        if pid == current_pid:
                            process_gpu_mem = used_mem_val
                            break
        except Exception:
            pass
        gpu_info_str = f"显存占比: {gpu_percent:.1f}% ({used_gpu:.0f}/{total_gpu:.0f} MB), 本进程显存: {process_gpu_mem:.0f} MB"
    except Exception:
        pass
        
    print(f"[{stage_name}] 内存占比: {mem.percent:.1f}% ({used_mem:.2f}/{total_mem:.2f} GB), 本进程内存: {process_mem:.1f} MB | {gpu_info_str}")

def main(args):
    # 1. 路径与配置
    print_system_status("初始化前")
    init_start = time.time()
    model_dir = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/"
    config_path = os.path.join(model_dir, "config.yaml")
    if args.speech == "INT8":
        speech_onnx_name = "model_speech_int8.onnx"
    elif args.speech == "INT4":
        speech_onnx_name = "model_speech_int4.onnx"
    else:
        speech_onnx_name = "model_speech.onnx"
    speech_onnx_path = os.path.join(model_dir, speech_onnx_name)
    llm_onnx_path = os.path.join(model_dir, "model_llm.onnx")
    embed_path = os.path.join(model_dir, "embed_tokens.npy")
    
    test_wav = args.wav
    
    # 2. 载入词嵌入权重和 Tokenizer
    print("载入 LLM 词嵌入权重和 Tokenizer...")
    embed_tokens = np.load(embed_path)
    
    if str(embed_tokens.dtype).startswith('void') or embed_tokens.dtype.kind == 'V':
        import ml_dtypes
        embed_tokens = embed_tokens.view('bfloat16')
        
    if args.llm == "BF16":
        import ml_dtypes
        infer_dtype = 'bfloat16'
        if embed_tokens.dtype != ml_dtypes.bfloat16:
            embed_tokens = embed_tokens.astype('bfloat16')
        io_dtype = np.float32
        print("【模式】大语言模型端使用 Bfloat16 精度进行推理...")
    elif args.llm == "FP16":
        infer_dtype = np.float16
        embed_tokens = embed_tokens.astype(np.float16)
        io_dtype = np.float16
        print("【模式】大语言模型端使用 Float16 精度进行推理...")
    else:
        infer_dtype = np.float32
        embed_tokens = embed_tokens.astype(np.float32)
        io_dtype = np.float32
        print("【模式】大语言模型端使用 Float32 精度进行推理...")
        
    tokenizer = AutoTokenizer.from_pretrained("/home/inno/code_inno/innoreport/ASR/Qwen3-0.6B")
    
    # 载入热词列表
    try:
        with open('/media/inno/ASR/胃镜/gi_hotwords_v3.txt', "r", encoding="utf-8") as f:
            hotword_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("未找到热词文件，将退回为空热词列表。")
        hotword_list = []
    
    # 3. 初始化 WavFrontend 并提取特征
    print("正在载入 WavFrontend 配置并提取语音特征...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    frontend_conf = config["frontend_conf"]
    frontend = WavFrontend(**frontend_conf)
    
    waveform, sample_rate = torchaudio.load(test_wav)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    speech, speech_lengths = frontend(waveform, [waveform.shape[1]])
    speech = speech.numpy()
    speech_lengths = np.array(speech_lengths, dtype=np.int32)
    
    # 4. 常驻初始化推理会话
    print("开始初始化 ONNX 推理会话...")
    
    # 【极致显存优化点】：声学端强制在 CPU 上运行，完全归零其在 GPU 上的显存开销 (包括 CUDA 上下文和 Arena)
    print("1/2. 初始化声学端推理会话 (强制 CPU 运行) ...")
    speech_session = ort.InferenceSession(speech_onnx_path, providers=["CPUExecutionProvider"])
    
    # 大语言模型在 GPU 上运行，启用 I/O 绑定
    print("2/2. 初始化大语言模型 (LLM) 推理会话 (GPU 运行，启用紧凑显存策略) ...")
    cuda_provider_opts = {
        "device_id": "0",
        "arena_extend_strategy": "kSameAsRequested",
        "do_copy_in_default_stream": "True"
    }
    providers = [
        ("CUDAExecutionProvider", cuda_provider_opts),
        "CPUExecutionProvider"
    ]
    
    llm_opts = ort.SessionOptions()
    # 强制 Initializers (权重) 直接使用 GPU 设备分配器，消除冗余拷贝和堆碎片
    llm_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    
    if args.llm == "BF16":
        print("【BF16 优化】将大模型图优化级别设置为 ORT_ENABLE_BASIC 以规避 QuickGelu 融合限制...")
        llm_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
    llm_session = ort.InferenceSession(llm_onnx_path, sess_options=llm_opts, providers=providers)
    
    # 首次修剪 CPU 堆内存，使常驻状态最纯净
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
        
    init_time = time.time() - init_start
    print(f"\n>>> [模型常驻初始化耗时]: {init_time:.3f} 秒")
    print_system_status("初始化后")
    
    # 5. 执行声学端推理，获取语音适配表征 (在 CPU 运行，极快且 0 显存)
    print("\n运行声学前端 ONNX 提取语音表征 (CPU 运行)...")
    inference_start = time.time()
    encoder_out, encoder_out_lens = speech_session.run(
        ["encoder_out", "encoder_out_lens"], 
        {"speech": speech, "speech_lengths": speech_lengths}
    )
    speech_infer_time = time.time() - inference_start
    print(f"声学特征提取完成，耗时: {speech_infer_time:.3f} 秒")
    
    # 6. 构建 Prompt 并拼接输入 embeddings
    hotwords_str = ", ".join(hotword_list)
    prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n热词列表：[{hotwords_str}]\n语音转写："
    
    prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}"
    suffix = f"<|im_end|>\n<|im_start|>assistant\n"
    
    prefix_ids = tokenizer.encode(prefix)
    suffix_ids = tokenizer.encode(suffix)
    
    prefix_embeds = embed_tokens[prefix_ids]
    suffix_embeds = embed_tokens[suffix_ids]
    
    speech_len = int(encoder_out_lens[0])
    speech_token = np.array(encoder_out[0, :speech_len, :], dtype=infer_dtype)
    
    inputs_embeds = np.concatenate([prefix_embeds, speech_token, suffix_embeds], axis=0)
    inputs_embeds = np.expand_dims(inputs_embeds, axis=0).astype(io_dtype)
    
    attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int32)
    
    # 7. 自回归解码循环 (ONNX I/O Binding)
    print("启动 LLM 自回归解码循环 (ONNX I/O Binding)...")
    llm_start = time.time()
    
    num_layers = 28
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        
    io_binding = llm_session.io_binding()
    
    inputs_embeds_val = ort.OrtValue.ortvalue_from_numpy(inputs_embeds, 'cuda', 0)
    attention_mask_val = ort.OrtValue.ortvalue_from_numpy(attention_mask, 'cuda', 0)
    position_ids = np.arange(inputs_embeds.shape[1], dtype=np.int64).reshape(1, -1)
    position_ids_val = ort.OrtValue.ortvalue_from_numpy(position_ids, 'cuda', 0)
    
    io_binding.bind_ortvalue_input("inputs_embeds", inputs_embeds_val)
    io_binding.bind_ortvalue_input("attention_mask", attention_mask_val)
    io_binding.bind_ortvalue_input("position_ids", position_ids_val)
    
    for i in range(num_layers):
        empty_kv = np.zeros((1, 8, 0, 128), dtype=io_dtype)
        k_val = ort.OrtValue.ortvalue_from_numpy(empty_kv, 'cuda', 0)
        v_val = ort.OrtValue.ortvalue_from_numpy(empty_kv, 'cuda', 0)
        
        io_binding.bind_ortvalue_input(f"past_key_{i}", k_val)
        io_binding.bind_ortvalue_input(f"past_value_{i}", v_val)
        
    for name in output_names:
        io_binding.bind_output(name, device_type='cuda', device_id=0)
        
    llm_session.run_with_iobinding(io_binding)
    outputs_val = io_binding.get_outputs()
    
    logits_numpy = outputs_val[0].numpy()
    next_token_id = int(np.argmax(logits_numpy[0, -1, :]))
    
    total_seq_len = inputs_embeds.shape[1] + 1
    attention_mask = np.ones((1, total_seq_len), dtype=np.int32)
    
    generated_ids = []
    max_new_tokens = 512
    
    if next_token_id == 151645:
        pass
    else:
        generated_ids.append(next_token_id)
        
        for step in range(1, max_new_tokens):
            next_embed = np.expand_dims(embed_tokens[next_token_id].astype(io_dtype), axis=(0, 1))
            next_embed_val = ort.OrtValue.ortvalue_from_numpy(next_embed, 'cuda', 0)
            
            step_position_ids = np.array([[total_seq_len - 1]], dtype=np.int64)
            step_position_ids_val = ort.OrtValue.ortvalue_from_numpy(step_position_ids, 'cuda', 0)
            attention_mask_val = ort.OrtValue.ortvalue_from_numpy(attention_mask, 'cuda', 0)
            
            step_io_binding = llm_session.io_binding()
            step_io_binding.bind_ortvalue_input("inputs_embeds", next_embed_val)
            step_io_binding.bind_ortvalue_input("attention_mask", attention_mask_val)
            step_io_binding.bind_ortvalue_input("position_ids", step_position_ids_val)
            
            for i in range(num_layers):
                step_io_binding.bind_ortvalue_input(f"past_key_{i}", outputs_val[1 + 2 * i])
                step_io_binding.bind_ortvalue_input(f"past_value_{i}", outputs_val[1 + 2 * i + 1])
                
            for name in output_names:
                step_io_binding.bind_output(name, device_type='cuda', device_id=0)
                
            llm_session.run_with_iobinding(step_io_binding)
            outputs_val = step_io_binding.get_outputs()
            
            logits_numpy = outputs_val[0].numpy()
            next_token_id = int(np.argmax(logits_numpy[0, -1, :]))
            
            if next_token_id == 151645:
                break
                
            generated_ids.append(next_token_id)
            total_seq_len += 1
            attention_mask = np.ones((1, total_seq_len), dtype=np.int32)
        
    llm_infer_time = time.time() - llm_start
    total_infer_time = time.time() - inference_start
    
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("=" * 60)
    print(f"【高性能常驻 ONNX 自回归解码结果】:\n{decoded_text}")
    print("=" * 60)
    print(f"\n>>> [推理性能统计]:")
    print(f"  - 声学前端推理耗时: {speech_infer_time:.3f} 秒")
    print(f"  - 大语言模型自回归推理耗时: {llm_infer_time:.3f} 秒 (生成 {len(generated_ids)} 个 token, 平均 {llm_infer_time/len(generated_ids):.4f} 秒/token)")
    print(f"  - 端到端推理总耗时: {total_infer_time:.3f} 秒")
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    print_system_status("推理后")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="常驻低显存高性能推理脚本")
    parser.add_argument(
        "--speech",
        type=str,
        default="INT8",
        choices=["FP32", "INT8", "INT4", "fp32", "int8", "int4"],
        help="声学端模式，支持 FP32/INT8/INT4，默认为 FP32"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="FP16",
        choices=["FP16", "BF16", "FP32", "fp16", "bf16", "fp32"],
        help="大语言模型端精度参数，默认为 FP32"
    )
    parser.add_argument(
        "--wav",
        type=str,
        # default="/media/inno/ASR/胃镜/audio/test/LLM_V5_test/171_5955.mp3",
        # default="/media/inno/ASR/胃镜/audio/test/zxs/test.mp3",
        default="/media/inno/ASR/胃镜/audio/test/0601/test.mp3",
        help="输入音频文件的路径，默认为测试音频"
    )
    args = parser.parse_args()
    args.speech = args.speech.upper()
    args.llm = args.llm.upper()
    main(args)
