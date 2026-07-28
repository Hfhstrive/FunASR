import time
import os
import sys
import glob
import json
import argparse
import subprocess
import tempfile
import psutil

# 初始化 psutil 采样器基准线
psutil.cpu_percent(interval=None)
psutil.Process(os.getpid()).cpu_percent(interval=None)

# 动态引入 asr-hotword 项目
sys.path.append('/home/inno/code/ASR/asr-hotword')
try:
    from hotword import PhonemeCorrector
except ImportError:
    print("警告：未能加载 PhonemeCorrector，之后将仅进行原始 ASR 推理。")
    PhonemeCorrector = None


def get_resource_usage():
    usage = {}
    usage["cpu_percent"] = psutil.cpu_percent(interval=None)
    process = psutil.Process(os.getpid())
    usage["process_cpu_percent"] = process.cpu_percent(interval=None)
    return usage


def run_llama_funasr_cli(cli_bin: str, enc_gguf: str, llm_gguf: str, audio_path: str, chunk_sec: float = 15.0, enc_device: str = "gpu", prompt_text: str = "语音转写成中文：") -> str:
    """
    通过 C++ 原生 llama-funasr-cli 执行声学编码器 (funasr-encoder-f16.gguf) + LLM (qwen3-0.6b-q8_0.gguf) 端到端 GGUF 推理
    """
    # 统一将音频/视频文件转换成标准 16kHz 16-bit 单声道 WAV 文件
    tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav = tf.name
    tf.close()

    try:
        ffmpeg_cmd = f'ffmpeg -i "{audio_path}" -acodec pcm_s16le -ar 16000 -ac 1 -y "{temp_wav}" > /dev/null 2>&1'
        os.system(ffmpeg_cmd)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + env.get("LD_LIBRARY_PATH", "")

        cmd = [
            cli_bin,
            "--enc", enc_gguf,
            "-m", llm_gguf,
            "-a", temp_wav,
            "--enc-device", enc_device,
            "--prompt", prompt_text,
            "--chunk", str(chunk_sec)
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        lines = res.stdout.strip().split('\n')
        # 获取最后一行生成的识别文本
        raw_text = lines[-1].strip() if lines else ""
        return raw_text
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


def find_media_files(target_path: str):
    """
    递归检索目录下的所有常见音视频媒体文件
    """
    valid_exts = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.mp4', '.mkv'}
    media_files = []

    if os.path.isfile(target_path):
        ext = os.path.splitext(target_path)[1].lower()
        if ext in valid_exts:
            media_files.append(target_path)
    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    media_files.append(os.path.join(root, file))
    
    return sorted(media_files)


def main():
    parser = argparse.ArgumentParser(description="纯 GGUF (funasr-encoder + qwen3) C++ 绑定批量推理与纠错脚本")
    parser.add_argument(
        "--cli_bin",
        type=str,
        default="/home/inno/code/LLM/llama.cpp/build/bin/llama-funasr-cli",
        help="llama-funasr-cli 可执行文件路径"
    )
    parser.add_argument(
        "--enc_gguf",
        type=str,
        default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/gguf/funasr-encoder-f16.gguf",
        help="GGUF 声学编码器文件路径 (funasr-encoder-f16.gguf)"
    )
    parser.add_argument(
        "--llm_gguf",
        type=str,
        default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/gguf/qwen3-0.6b-q8_0.gguf",
        help="GGUF LLM 文件路径 (qwen3-0.6b-q8_0.gguf)"
    )
    parser.add_argument(
        "--enc_device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="声学编码器 (funasr-encoder-f16.gguf) 计算设备: gpu 或 cpu"
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="语音转写成中文：",
        help="Prompt 提示词文本 (默认: 语音转写成中文：)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="待批量处理的音视频文件夹或具体文件路径 (若不填则使用默认测试列表)"
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
    args = parser.parse_args()

    # 校验组件文件路径
    if not os.path.exists(args.cli_bin):
        raise FileNotFoundError(f"未找到 llama-funasr-cli: {args.cli_bin}")
    if not os.path.exists(args.enc_gguf):
        raise FileNotFoundError(f"未找到 enc_gguf: {args.enc_gguf}")
    if not os.path.exists(args.llm_gguf):
        raise FileNotFoundError(f"未找到 llm_gguf: {args.llm_gguf}")

    print("=" * 60)
    print(" 正在加载纯 GGUF 双组件 (funasr-encoder-f16.gguf + qwen3-0.6b-q8_0.gguf)...")
    print(f"- 声学编码器: {args.enc_gguf}")
    print(f"- 解码大模型: {args.llm_gguf}")

    # 1. 收集待批量处理的媒体文件
    audio_paths = []
    if args.audio_dir and os.path.exists(args.audio_dir):
        audio_paths = find_media_files(args.audio_dir)
        print(f"在目录 [{args.audio_dir}] 中扫描到 {len(audio_paths)} 个媒体文件。")
    
    if not audio_paths:
        default_paths = [
            '/media/inno/ASR/胃镜/input/test2.wav',
            '/media/inno/ASR/胃镜/audio/test/zxs/test.mp3',
            '/media/inno/ASR/胃镜/audio/test/0601/test.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/胃癌.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/反流性食管炎.mp3',
            '/media/inno/ASR/胃镜/audio/test/hfh/萎缩肠化.mp3',
        ]
        audio_paths = [p for p in default_paths if os.path.exists(p)]
        print(f"使用内置基准测试列表，共计 {len(audio_paths)} 个音频文件。")

    # 2. 初始化后处理纠错器
    corrector = None
    if PhonemeCorrector is not None and os.path.exists(args.hotword_txt_path):
        corrector = PhonemeCorrector(threshold=0.85)
        with open(args.hotword_txt_path, "r", encoding="utf-8") as f:
            corrector.update_hotwords(f.read())
        print("后处理医学纠错词表加载完成。")

    # 3. 结果保存配置
    os.makedirs(args.result_save_dir, exist_ok=True)
    results_json_path = os.path.join(args.result_save_dir, "batch_results_pure_gguf.json")
    results_dict = {}

    psutil.cpu_percent(interval=None)
    psutil.Process(os.getpid()).cpu_percent(interval=None)
    total_infer_start = time.time()

    print("=" * 60)
    print(f"开始执行批量 GGUF 推理，共 {len(audio_paths)} 个文件...")

    for idx, audio_path in enumerate(audio_paths, 1):
        filename = os.path.basename(audio_path)
        infer_single_start = time.time()

        raw_text = run_llama_funasr_cli(
            cli_bin=args.cli_bin,
            enc_gguf=args.enc_gguf,
            llm_gguf=args.llm_gguf,
            audio_path=audio_path,
            chunk_sec=15.0,
            enc_device=args.enc_device,
            prompt_text=args.prompt_text
        )

        # 后处理纠错
        if corrector is not None:
            corrected_res = corrector.correct(raw_text)
            final_text = corrected_res.text
            match_info = corrected_res.matches
        else:
            final_text = raw_text
            match_info = None

        infer_single_time = time.time() - infer_single_start

        print("-" * 50)
        print(f"[{idx}/{len(audio_paths)}] 音频: {filename} (耗时: {infer_single_time:.4f} 秒)")
        print(f"原始识别: {raw_text}")
        print(f"纠错结果: {final_text}")
        if match_info:
            print(f"纠错替换: {match_info}")

        case_key = f"{idx:04d}_{filename}"
        results_dict[case_key] = {
            "file_path": audio_path,
            "infer_time_sec": round(infer_single_time, 4),
            "asr_result": raw_text,
            "corrected_result": final_text,
            "matches": match_info
        }

    total_infer_time = time.time() - total_infer_start
    usage_after_infer = get_resource_usage()

    print("\n" + "=" * 60)
    print(" 纯 GGUF 批量推理性能统计结果 ".center(60, "="))
    print(f"- 批量处理总文件数: {len(audio_paths)}")
    print(f"- 总推理运行时间: {total_infer_time:.4f} 秒")
    if len(audio_paths) > 0:
        print(f"- 单句平均推理耗时: {(total_infer_time / len(audio_paths)):.4f} 秒/样本")
    print(f"- 系统总 CPU 占比: {usage_after_infer['cpu_percent']:.2f}%")

    with open(results_json_path, "w", encoding="utf-8") as fj:
        json.dump(results_dict, fj, ensure_ascii=False, indent=4)

    print(f"- 批量推理结果已保存至 JSON: {results_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
