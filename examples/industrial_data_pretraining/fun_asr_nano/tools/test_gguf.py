import time
import os
import sys
import glob
import json
import ctypes
import argparse
import tempfile
import psutil
import torch

# 动态确保包含 CUDA 运行时库路径
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + os.environ.get("LD_LIBRARY_PATH", "")

# 动态添加依赖环境 site-packages 路径
sys.path.append('/home/inno/anaconda3/envs/llama/lib/python3.12/site-packages')

try:
    from funasr import AutoModel
    from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
except ImportError as e:
    print(f"[错误] 无法导入 FunASR 模块: {e}")
    sys.exit(1)

try:
    from llama_cpp import Llama, llama_cpp
except ImportError as e:
    print(f"[错误] 无法导入 llama_cpp 模块: {e}")
    sys.exit(1)

# 初始化 psutil 采样器基准线
psutil.cpu_percent(interval=None)
psutil.Process(os.getpid()).cpu_percent(interval=None)


def get_resource_usage(device="cuda:0"):
    usage = {}
    usage["cpu_percent"] = psutil.cpu_percent(interval=None)
    process = psutil.Process(os.getpid())
    usage["process_cpu_percent"] = process.cpu_percent(interval=None)

    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            used_mem = total_mem - free_mem
            usage["gpu_total_mb"] = total_mem / 1024 / 1024
            usage["gpu_used_mb"] = used_mem / 1024 / 1024
            usage["gpu_percent"] = (used_mem / total_mem) * 100

            torch_allocated = torch.cuda.memory_allocated(device)
            usage["torch_allocated_mb"] = torch_allocated / 1024 / 1024
            usage["torch_percent"] = (torch_allocated / total_mem) * 100
        except Exception as e:
            usage["gpu_error"] = str(e)
    return usage


# 动态引入 asr-hotword 项目
sys.path.append('/home/inno/code/ASR/asr-hotword')
try:
    from hotword import PhonemeCorrector
except ImportError:
    print("警告：未能加载 PhonemeCorrector，之后将仅进行原始 ASR 推理。")
    PhonemeCorrector = None


class FunASRNanoGGUFEngine:
    """
    Fun-ASR-Nano 结合 GGUF LLM 的混合推理引擎：
    - 声学前端、SenseVoiceEncoder 与 AudioAdaptor 由 PyTorch 在 GPU 上高效编码
    - LLM 解码部分使用从微调导出的 Qwen3-0.6B GGUF 量化模型 (llama_cpp)
    """

    def __init__(self, finetune_dir: str, gguf_model_path: str, enc_device: str = "cpu", n_ctx: int = 4096):
        self.finetune_dir = finetune_dir
        self.gguf_model_path = gguf_model_path
        self.enc_device = enc_device

        if not os.path.exists(finetune_dir):
            raise FileNotFoundError(f"微调模型目录不存在: {finetune_dir}")
        if not os.path.exists(gguf_model_path):
            raise FileNotFoundError(f"GGUF 模型文件不存在: {gguf_model_path}")

        print(f"1. 正在加载 Fun-ASR-Nano PyTorch 声学模块 (运行于 {enc_device}): {finetune_dir}")
        self.funasr_model = AutoModel(
            model=finetune_dir,
            device=enc_device,
            disable_update=True,
            dtype="fp32" if enc_device == "cpu" else "fp16",
        )
        self.core_model = (
            self.funasr_model.model[0]
            if isinstance(self.funasr_model.model, list)
            else self.funasr_model.model
        )
        self.frontend = self.funasr_model.kwargs['frontend']
        self.audio_encoder = self.core_model.audio_encoder
        self.audio_adaptor = self.core_model.audio_adaptor
        self.embed_tokens = self.core_model.llm.model.embed_tokens
        self.tokenizer = self.funasr_model.kwargs['tokenizer']

        # 释放重复的 PyTorch LLM Backbone 显存
        if hasattr(self.core_model, 'llm'):
            del self.core_model.llm
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"2. 正在加载 GGUF LLM 模型 (llama_cpp 运行于 GPU): {gguf_model_path}")
        self.llm = Llama(
            model_path=gguf_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )

    def transcribe(self, wav_path: str, prompt_text: str = "语音转写成中文：", max_tokens: int = 256) -> str:
        # A. 声学特征提取与 Adaptor 映射 (在 enc_device CPU/GPU 上高效计算)
        speech_data = load_audio_text_image_video(wav_path, fs=self.frontend.fs)
        speech, speech_lengths = extract_fbank([speech_data], data_type='sound', frontend=self.frontend, is_final=True)
        speech = speech.to(self.enc_device, dtype=torch.float32)
        speech_lengths = speech_lengths.to(self.enc_device)

        with torch.no_grad():
            enc_out, enc_lens = self.audio_encoder(speech, speech_lengths)
            adp_dtype = next(self.audio_adaptor.parameters()).dtype
            adp_out, adp_lens = self.audio_adaptor(enc_out.to(dtype=adp_dtype), enc_lens)

            # 计算 Low Frame Rate 限制下的生效 Token 长度
            fbank_len = speech_lengths[0].item()
            olens = 1 + (fbank_len - 3 + 2 * 1) // 2
            olens = 1 + (olens - 3 + 2 * 1) // 2
            fake_token_len = (olens - 1) // 2 + 1

            audio_emb = adp_out[0, :fake_token_len, :]

            # B. 拼接 ChatML Prompt Embedding
            prefix_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}"
            suffix_text = "<|im_end|>\n<|im_start|>assistant\n"

            prefix_ids = self.tokenizer.encode(prefix_text, allowed_special='all')
            suffix_ids = self.tokenizer.encode(suffix_text, allowed_special='all')

            prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.enc_device))
            suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.enc_device))

            input_embeds = torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0).float()

        # C. 将 Embedding 拷贝至 GGUF llama_batch 并进行上下文评估
        n_tokens = input_embeds.shape[0]
        flat_embeds = input_embeds.cpu().contiguous().view(-1).numpy()

        batch = llama_cpp.llama_batch_init(n_tokens, 1024, 1)
        try:
            batch.n_tokens = n_tokens
            for i in range(n_tokens):
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = (i == n_tokens - 1)

            ctypes.memmove(batch.embd, flat_embeds.ctypes.data, n_tokens * 1024 * 4)

            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            decode_res = llama_cpp.llama_decode(self.llm._ctx.ctx, batch)
            if decode_res != 0:
                raise RuntimeError(f"GGUF 解码评估失败，错误码: {decode_res}")
            self.llm.n_tokens = n_tokens

            # D. 基于评估后的 KV Cache 进行逐 Token 采样解码
            generated_tokens = []
            eos_tokens = {self.llm.token_eos(), 151645, 151643}

            for _ in range(max_tokens):
                token_id = self.llm.sample(temp=0.0)
                if token_id in eos_tokens:
                    break
                generated_tokens.append(token_id)
                self.llm.eval([token_id])

            raw_text = self.llm.detokenize(generated_tokens).decode('utf-8', errors='ignore')
            return raw_text.strip()
        finally:
            llama_cpp.llama_batch_free(batch)


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano GGUF 模型推理与纠错脚本")
    parser.add_argument(
        "--finetune_dir",
        type=str,
        default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/",
        help="微调模型所在的目录"
    )
    parser.add_argument(
        "--gguf_path",
        type=str,
        default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/gguf/qwen3-0.6b-q8_0.gguf",
        help="GGUF 模型文件路径 (例如 qwen3-0.6b-q8_0.gguf 或 qwen3-0.6b-f32.gguf)"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/media/inno/ASR/ChatML/V4/val",
        help="评估音频所在根目录"
    )
    parser.add_argument(
        "--hotword_txt_path",
        type=str,
        default="/media/inno/ASR/gi_hotwords.txt",
        help="热词纠错词表路径"
    )
    parser.add_argument(
        "--result_save_dir",
        type=str,
        default="/media/inno/output/ASR/ChatML/V4/val/",
        help="结果保存路径"
    )
    parser.add_argument(
        "--enc_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda:0", "cuda"],
        help="声学编码器计算设备 (设为 cpu 显著降低显存占比)"
    )
    args = parser.parse_args()

    # 1. 初始化 Fun-ASR-Nano GGUF 混合推理引擎
    print("=" * 60)
    print("正在初始化 Fun-ASR-Nano (GGUF) 混合推理引擎...")
    init_start_time = time.time()

    engine = FunASRNanoGGUFEngine(
        finetune_dir=args.finetune_dir,
        gguf_model_path=args.gguf_path,
        enc_device=args.enc_device,
        n_ctx=4096,
    )

    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    usage_after_init = get_resource_usage(device="cuda:0")
    print(f"ASR GGUF 模型初始化耗时: {init_time:.4f} 秒")
    if "gpu_percent" in usage_after_init:
        print(f"初始化后显存占比 (系统整体): {usage_after_init['gpu_percent']:.2f}% ({usage_after_init['gpu_used_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")
        print(f"初始化后显存占比 (PyTorch分配): {usage_after_init['torch_percent']:.2f}% ({usage_after_init['torch_allocated_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")

    # 2. 初始化音素级后处理纠错器
    print("-" * 50)
    print("正在加载后处理医学纠错词表...")
    corrector = None
    if PhonemeCorrector is not None:
        corrector = PhonemeCorrector(threshold=0.85)
        if os.path.exists(args.hotword_txt_path):
            with open(args.hotword_txt_path, "r", encoding="utf-8") as f:
                hotwords_content = f.read()
            corrector.update_hotwords(hotwords_content)
            print("纠错词表加载完成。")
        else:
            print(f"警告：未找到热词文件 {args.hotword_txt_path}")

    # 3. 收集待测音频列表 (与 test.py 完全对齐)
    # audio_paths = glob.glob(f'{args.val_path}/src0/*.wav') + glob.glob(f'{args.val_path}/src1/*.wav')
    audio_paths = None
    if not audio_paths:
        print(f"在 {args.val_path} 下未检索到 src0/src1 wav 文件，切换为后备音频列表。")
        audio_paths = [
            '/media/inno/ASR/胃镜/input/test2.wav',
            '/media/inno/ASR/胃镜/audio/test/zxs/test.mp3',
            '/media/inno/ASR/胃镜/audio/test/0601/test.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/胃癌.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/反流性食管炎.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/萎缩肠化.mp3',
        ]
        audio_paths = [p for p in audio_paths if os.path.exists(p)]

    print(f"总计收集到待测音频数: {len(audio_paths)}")

    # 4. 结果保存配置
    os.makedirs(args.result_save_dir, exist_ok=True)
    results_json_path = os.path.join(args.result_save_dir, "val_results_gguf.json")
    results_dict = {}

    # 重置 psutil 计时基准
    psutil.cpu_percent(interval=None)
    psutil.Process(os.getpid()).cpu_percent(interval=None)
    total_infer_start = time.time()

    # 5. 循环执行 GGUF 推理与音素纠错
    for audio_path in audio_paths:
        audio_type = audio_path.split('.')[-1].lower()
        temp_wav = None

        if audio_type != 'wav':
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav = temp_file.name
            temp_file.close()
            cmd = f'ffmpeg -i "{audio_path}" -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_wav}" > /dev/null 2>&1'
            os.system(cmd)
            infer_target_path = temp_wav
        else:
            infer_target_path = audio_path

        infer_single_start = time.time()
        try:
            raw_text = engine.transcribe(wav_path=infer_target_path)
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

        # 执行音素级纠错后处理
        if corrector is not None:
            corrected_res = corrector.correct(raw_text)
            final_text = corrected_res.text
            match_info = corrected_res.matches
        else:
            final_text = raw_text
            match_info = None

        infer_single_time = time.time() - infer_single_start

        print("-" * 50)
        print(f"音频: {os.path.basename(audio_path)} (推理+纠错单句耗时: {infer_single_time:.4f} 秒)")
        print(f"原始识别: {raw_text}")
        print(f"纠错结果: {final_text}")
        if match_info:
            print(f"纠错替换: {match_info}")

        # 提取唯一的 case 键 (例如 src0_case-no) 并存入字典
        parts = audio_path.split('/')
        src_name = parts[-2] if len(parts) > 1 else "src0"
        case_name = os.path.basename(audio_path).replace(f".{audio_type}", '')
        case_key = f"{src_name}_{case_name}"

        results_dict[case_key] = {
            "asr_result": raw_text,
            "corrected_result": final_text
        }

    total_infer_time = time.time() - total_infer_start

    # 6. 获取推理结束时的资源占用与性能统计
    usage_after_infer = get_resource_usage(device="cuda:0")
    cpu_infer_system = psutil.cpu_percent(interval=None)
    cpu_infer_process = psutil.Process(os.getpid()).cpu_percent(interval=None)
    cpu_count = psutil.cpu_count() or 1
    process_cpu_ratio = cpu_infer_process / cpu_count

    print("\n" + "=" * 60)
    print(" Fun-ASR-Nano GGUF 性能统计结果 ".center(60, "="))
    print(f"- 模型初始化时间: {init_time:.4f} 秒")
    print(f"- 总推理运行时间: {total_infer_time:.4f} 秒")
    if len(audio_paths) > 0:
        print(f"- 单句平均推理耗时: {(total_infer_time / len(audio_paths)):.4f} 秒/样本")
    if "gpu_percent" in usage_after_infer:
        print(f"- 推理后显存占比 (系统整体): {usage_after_infer['gpu_percent']:.2f}% ({usage_after_infer['gpu_used_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
        print(f"- 推理后显存占比 (PyTorch分配): {usage_after_infer['torch_percent']:.2f}% ({usage_after_infer['torch_allocated_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
    print(f"- 进程 CPU 占比: {cpu_infer_process:.2f}% (折算系统整体占用为: {process_cpu_ratio:.2f}%)")
    print(f"- 系统总 CPU 占比: {cpu_infer_system:.2f}%")

    with open(results_json_path, "w", encoding="utf-8") as fj:
        json.dump(results_dict, fj, ensure_ascii=False, indent=4)

    print(f"- 推理结果已统一保存至 JSON: {results_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
