#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实录音的工业级阳性唤醒词声学离线数据增强脚本 (Data Augmentation)
运行环境: torch2.7.1 (/home/inno/anaconda3/envs/torch2.7.1/bin/python)

针对输入:
  /media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实 (160 个真实音频)
输出目标:
  /media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实_增强

增强策略 (每个音频衍生 8 种高保真声学变体):
1. orig         : 原速规整化 (统一 16000Hz, 单声道, 16-bit)
2. sp0.9        : 慢速 0.9x (模拟沉稳/慢速拉长音)
3. sp1.1        : 快速 1.1x (模拟急促/快速吐字)
4. vol0.6       : 远场弱音 (0.6x 音量，模拟站位远离麦克风)
5. vol1.3       : 近场强音 (1.3x 音量，模拟靠近麦克风大声说话)
6. pitch_up     : 升高 1 个半音 (+100 cents，模拟高亢声调/女性/偏亮声线)
7. pitch_down   : 降低 1 个半音 (-100 cents，模拟低沉声调/深沉男声)
8. noise_snr20  : 叠加 MUSAN 真实环境底噪 (轻度, SNR 20dB)
9. noise_snr15  : 叠加 MUSAN 真实环境底噪 (中度, SNR 15dB)
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path

import torch
import torchaudio


def load_and_resample(wav_path, target_sr=16000):
    """加载音频并统一转换为单声道与 16000Hz 采样率"""
    waveform, sr = torchaudio.load(wav_path)
    # 转单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # 重采样为 target_sr
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def apply_speed(waveform, sr, speed=0.9):
    """变速扰动 (保持音调)"""
    effects = [["speed", str(speed)], ["rate", str(sr)]]
    augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
    return augmented


def apply_pitch(waveform, sr, n_cents=100):
    """变调扰动 (100 cents = 1 半音)"""
    effects = [["pitch", str(n_cents)], ["rate", str(sr)]]
    augmented, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
    return augmented


def apply_volume(waveform, gain=1.2):
    """音量/增益扰动 (带防爆音限幅保护)"""
    augmented = waveform * gain
    max_val = torch.max(torch.abs(augmented))
    if max_val > 0.99:
        augmented = augmented / max_val * 0.98
    return augmented


def add_noise(clean_waveform, noise_waveform, snr_db=15):
    """向干净语音叠加环境噪声，按指定信噪比 (SNR) 混合"""
    # 裁剪或循环扩展 noise 到与 clean 相同长度
    clean_len = clean_waveform.shape[1]
    noise_len = noise_waveform.shape[1]

    if noise_len < clean_len:
        repeat_times = (clean_len // noise_len) + 1
        noise_waveform = noise_waveform.repeat(1, repeat_times)[:, :clean_len]
    elif noise_len > clean_len:
        start_idx = random.randint(0, noise_len - clean_len)
        noise_waveform = noise_waveform[:, start_idx : start_idx + clean_len]

    # 计算能量与缩放因子
    clean_power = clean_waveform.norm(p=2) ** 2 / clean_len
    noise_power = noise_waveform.norm(p=2) ** 2 / clean_len
    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (snr * noise_power + 1e-8))

    noisy = clean_waveform + scale * noise_waveform
    max_val = torch.max(torch.abs(noisy))
    if max_val > 0.99:
        noisy = noisy / max_val * 0.98
    return noisy


def collect_musan_noise_files(noise_dir):
    """扫描 MUSAN 噪声库中的所有 WAV 文件"""
    if not os.path.exists(noise_dir):
        return []
    p = Path(noise_dir)
    return list(p.rglob("*.wav"))


def main():
    parser = argparse.ArgumentParser(description="KWS 真实阳性样本声学离线数据增强")
    parser.add_argument("--input_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实",
                        help="原始真实正样本目录")
    parser.add_argument("--output_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实_增强",
                        help="增强后的音频输出目录")
    parser.add_argument("--musan_noise_dir", type=str, default="/media/inno/ASR/KWS/数据集/musan/musan/noise",
                        help="MUSAN 噪声集目录")
    parser.add_argument("--target_text", type=str, default="鹰眼鹰眼",
                        help="标注文本 (默认: 鹰眼鹰眼)")
    args = parser.parse_args()

    input_p = Path(args.input_dir)
    output_p = Path(args.output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    # 1. 扫描所有真实音频
    valid_exts = [".wav", ".mp3", ".m4a", ".flac"]
    all_raw_files = [f for f in input_p.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    all_raw_files.sort()

    print("\n" + "=" * 65)
    print(" KWS 真实阳性样本批量离线数据增强 ".center(65, "="))
    print(f"• 输入源目录   : {args.input_dir}")
    print(f"• 原始音频数   : {len(all_raw_files)} 个")
    print(f"• 输出目标目录 : {args.output_dir}")
    print(f"• 单样本扩展率 : 1 -> 9 (原规整 + 8 种声学扰动)")
    print(f"• 预计产出总数 : ~{len(all_raw_files) * 9} 个")
    print("=" * 65 + "\n")

    if not all_raw_files:
        print("[!] 错误: 输入目录下未找到有效音频文件！")
        sys.exit(1)

    # 2. 预加载/扫描 MUSAN 噪声集
    musan_noise_files = collect_musan_noise_files(args.musan_noise_dir)
    print(f"[*] 检测到 MUSAN 真实环境噪声文件共: {len(musan_noise_files)} 个")

    meta_file = output_p / "metadata.txt"
    wav_scp_file = output_p / "wav.scp"

    count_total = 0
    with open(meta_file, "w", encoding="utf-8") as f_meta, open(wav_scp_file, "w", encoding="utf-8") as f_scp:
        f_meta.write("# uttid\twav_path\ttext\tspk\taug_type\n")

        for idx, src_file in enumerate(all_raw_files):
            # 获取相对发音人子目录 (如 css/1.wav)
            rel_path = src_file.relative_to(input_p)
            spk = rel_path.parent.name if rel_path.parent.name else "default"
            stem = src_file.stem

            # 保持子目录结构
            out_spk_dir = output_p / spk
            out_spk_dir.mkdir(parents=True, exist_ok=True)

            try:
                clean_wf, sr = load_and_resample(str(src_file), target_sr=16000)
            except Exception as e:
                print(f"\n[!] 读取音频失败 {src_file}: {e}")
                continue

            # 9 种声学增强变体定义
            aug_variants = {}

            # 1. 原声规范化
            aug_variants["orig"] = clean_wf

            # 2. 慢速 0.9x
            try:
                aug_variants["sp0.9"] = apply_speed(clean_wf, sr, speed=0.9)
            except Exception:
                pass

            # 3. 快速 1.1x
            try:
                aug_variants["sp1.1"] = apply_speed(clean_wf, sr, speed=1.1)
            except Exception:
                pass

            # 4. 远场弱音 0.6x
            aug_variants["vol0.6"] = apply_volume(clean_wf, gain=0.6)

            # 5. 近场强音 1.3x
            aug_variants["vol1.3"] = apply_volume(clean_wf, gain=1.3)

            # 6. 音调微升 (+1 半音)
            try:
                aug_variants["pitch_up"] = apply_pitch(clean_wf, sr, n_cents=100)
            except Exception:
                pass

            # 7. 音调微降 (-1 半音)
            try:
                aug_variants["pitch_down"] = apply_pitch(clean_wf, sr, n_cents=-100)
            except Exception:
                pass

            # 8 & 9. 叠加真实环境底噪 (SNR 20dB & 15dB)
            if musan_noise_files:
                random_noise_path = random.choice(musan_noise_files)
                try:
                    noise_wf, _ = load_and_resample(str(random_noise_path), target_sr=16000)
                    aug_variants["noise_snr20"] = add_noise(clean_wf, noise_wf, snr_db=20)
                    aug_variants["noise_snr15"] = add_noise(clean_wf, noise_wf, snr_db=15)
                except Exception:
                    pass

            # 写入保存各个变体
            for aug_type, wf in aug_variants.items():
                utt_id = f"yy_{spk}_{stem}_{aug_type}"
                out_wav = out_spk_dir / f"{utt_id}.wav"
                torchaudio.save(str(out_wav), wf, sr, bits_per_sample=16)

                f_meta.write(f"{utt_id}\t{out_wav.resolve()}\t{args.target_text}\t{spk}\t{aug_type}\n")
                f_scp.write(f"{utt_id}\t{out_wav.resolve()}\n")
                count_total += 1

            sys.stdout.write(f"\r[{idx + 1}/{len(all_raw_files)}] 处理: {src_file.name} -> 生成 {len(aug_variants)} 种变体 (累计: {count_total})")
            sys.stdout.flush()

    print(f"\n\n[+] 离线数据增强全部完成！")
    print(f"[+] 原始输入音频 : {len(all_raw_files)} 个")
    print(f"[+] 产出增强音频 : {count_total} 个 (覆盖 9 种声学维度)")
    print(f"[+] 保存目录     : {args.output_dir}")
    print(f"[+] 索引文件     : {meta_file}")
    print(f"[+] 标准 wav.scp : {wav_scp_file}")


if __name__ == "__main__":
    main()
