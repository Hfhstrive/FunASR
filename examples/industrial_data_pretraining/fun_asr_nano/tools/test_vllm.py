#!/usr/bin/env python3
# coding: utf-8
"""
基于 vLLM 推理引擎 + ASR-Hotword 后处理纠错的极速医学推理脚本
"""

import os
import sys
import time
import torch
import psutil

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

# 1. 动态挂载 asr-hotword 纠错库
sys.path.append('/home/inno/code/ASR/asr-hotword')
try:
    from hotword import PhonemeCorrector
except ImportError as e:
    print(f"警告：导入纠错库失败。可能是路径错误或容器内缺少依赖（例如 pypinyin, rapidfuzz）。具体报错: {e}")
    PhonemeCorrector = None

from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

def main():
    # --- 配置区域 ---
    # 微调模型路径 (与 test.py 对齐)
    model_dir = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v3/"
    
    # 待测试的测试音频文件 (与 test.py 对齐)
    audio_paths = [
        '/media/inno/ASR/胃镜/input/test2.wav',
        '/media/inno/ASR/胃镜/audio/test/zxs/test.mp3',
        '/media/inno/ASR/胃镜/audio/test/0601/test.mp3',
        '/media/inno/ASR/胃镜/audio/test/hfh/胃癌.mp3',
        '/media/inno/ASR/胃镜/audio/test/hfh/反流性食管炎.mp3',
        '/media/inno/ASR/胃镜/audio/test/hfh/萎缩肠化.mp3',
    ]
    
    # 热词表路径 (与 test.py 对齐)
    hotword_txt_path = '/media/inno/ASR/gi_hotwords.txt'
    # 纠错阈值（0.85/0.8）
    corrector_threshold = 0.85
    # ----------------

    # 2. 初始化后处理纠错器
    corrector = None
    if PhonemeCorrector and os.path.exists(hotword_txt_path):
        print("正在加载后处理医学纠错词表...")
        corrector = PhonemeCorrector(threshold=corrector_threshold)
        with open(hotword_txt_path, "r", encoding="utf-8") as f:
            hotwords_content = f.read()
        corrector.update_hotwords(hotwords_content)
        print("纠错词表加载完成。")
    else:
        print(f"警告：未加载纠错库或未找到热词文件 {hotword_txt_path}，将跳过纠错流程。")

    # 对非 wav 格式音频进行预先转码 (与 test.py 对齐)
    processed_audio_paths = []
    for audio_path in audio_paths:
        audio_type = audio_path.split('.')[-1]
        if audio_type != 'wav':
            wav_path = audio_path.replace(audio_type, 'wav')
            if not os.path.exists(wav_path):
                os.system(f'ffmpeg -i {audio_path} -acodec pcm_s16le -ar 16000 -ac 1 -y {wav_path}')
            processed_audio_paths.append(wav_path)
        else:
            processed_audio_paths.append(audio_path)

    # 3. 初始化 vLLM 推理引擎
    print("\n正在初始化 vLLM 加速推理引擎并计时...")
    init_start_time = time.time()
    
    # 自动选择最适合的精度，防止 fp16 数值溢出导致模型崩溃输出大量的感叹号 "!"
    dtype = "fp32"
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                dtype = "bf16"
        except Exception:
            pass
    print(f"自动选用推理精度: {dtype}")

    engine = FunASRNanoVLLM.from_pretrained(
        model=model_dir,
        device="cuda:0",
        dtype=dtype,
        tensor_parallel_size=1,        # GPU 卡数配置
        gpu_memory_utilization=0.15,    # 显存占用上限
        max_model_len=2048,
        vllm_kwargs={"kv_cache_dtype": "fp8"} # 开启 KV 缓存 FP8 量化以减少显存占用
    )
    init_end_time = time.time()
    init_time = init_end_time - init_start_time

    usage_after_init = get_resource_usage(device="cuda:0")
    print(f"vLLM 引擎加载完成，耗时: {init_time:.4f}s")
    if "gpu_percent" in usage_after_init:
        print(f"初始化后显存占比 (系统整体): {usage_after_init['gpu_percent']:.2f}% ({usage_after_init['gpu_used_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")
        print(f"初始化后显存占比 (PyTorch分配): {usage_after_init['torch_percent']:.2f}% ({usage_after_init['torch_allocated_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")

    # 4. 执行推理与纠错
    print(f"\n开始转写并纠错 {len(processed_audio_paths)} 个音频文件...")
    
    # 重置 psutil 计时基准
    psutil.cpu_percent(interval=None)
    psutil.Process(os.getpid()).cpu_percent(interval=None)
    total_infer_start = time.time()
    
    # vLLM 支持批量推理，这里以当前音频列表前向
    results = engine.generate(
        inputs=processed_audio_paths,
        hotwords=None,      # 保持纯净解码状态，避免 LLM 前缀幻听
        language="中文",
        itn=True,           # 开启逆文本标准化
        max_new_tokens=512
    )
    
    # 5. 对比与后处理结果打印
    print("-" * 60)
    for r in results:
        audio_name = os.path.basename(r['key'])
        raw_text = r['text']
        
        # 执行纠错
        corrected_text = raw_text
        matches_detail = []
        if corrector:
            corr_res = corrector.correct(raw_text)
            corrected_text = corr_res.text
            matches_detail = corr_res.matches

        print(f"音频: {audio_name}")
        print(f"原始识别 (vLLM): {raw_text}")
        print(f"纠错结果: {corrected_text}")
        if matches_detail:
            print(f"纠错替换详情: {matches_detail}")
        else:
            print("纠错替换详情: (无替换)")
        print("-" * 60)

    total_infer_time = time.time() - total_infer_start

    # 获取推理结束时的资源占用
    usage_after_infer = get_resource_usage(device="cuda:0")
    # 获取推理期间的 CPU 占比（在此时间段内的平均值）
    cpu_infer_system = psutil.cpu_percent(interval=None)
    cpu_infer_process = psutil.Process(os.getpid()).cpu_percent(interval=None)
    cpu_count = psutil.cpu_count() or 1
    process_cpu_ratio = cpu_infer_process / cpu_count

    print("\n" + "=" * 60)
    print(" 性能统计结果 ".center(60, "="))
    print(f"- 模型初始化时间: {init_time:.4f} 秒")
    print(f"- 总推理运行时间: {total_infer_time:.4f} 秒")
    if "gpu_percent" in usage_after_infer:
        print(f"- 推理后显存占比 (系统整体): {usage_after_infer['gpu_percent']:.2f}% ({usage_after_infer['gpu_used_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
        print(f"- 推理后显存占比 (PyTorch分配): {usage_after_infer['torch_percent']:.2f}% ({usage_after_infer['torch_allocated_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
    print(f"- 进程 CPU 占比: {cpu_infer_process:.2f}% (折算系统整体占用为: {process_cpu_ratio:.2f}%)")
    print(f"- 系统总 CPU 占比: {cpu_infer_system:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
