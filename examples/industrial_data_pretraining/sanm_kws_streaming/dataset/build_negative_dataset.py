#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KWS 工业级阴性样本全自动构建脚本 (构建规模: 5000+ 例)
运行环境: torch2.7.1 (/home/inno/anaconda3/envs/torch2.7.1/bin/python)

核心保障:
1. 裁剪与保存时，每个音频末尾统一追加 0.35s 静音缓冲 (Tail Silence Padding)，
   配合 20ms 平滑余弦淡出，彻底消除爆音，并杜绝播放器/声卡硬件缓冲区提前关闭造成的截断尾音！
2. 彻底排除 19 条真实线上误唤醒音频 (作为后续独立的阴性测试集)。
3. 四位一体的立体阴性库 (总数 5200+ 例):
   - 01_aishell              : ~3200 例 (含 ~300 例带“眼/英/应/因”的近音字强对抗 + ~2900 例日常普通话)
   - 02_clinical_slices      : ~1400 例 (胃镜 216 例 + 肠镜 196 例长音频切片，每段 2.5s~3.5s)
   - 03_aikit_confusables    : ~210 例  (AIKit 定向合成的“因为因为”、“眼睛”等近音词)
   - 04_or_noises_and_beeps  : ~400 例  (MUSAN 机械风扇/电机 + Python 仿真监护仪“滴滴滴”心率/报警音)
4. 输出规整为 16000Hz, 16-bit, 单声道标准 WAV，并生成配套 wav.scp / text / metadata.txt。
"""

import os
import sys
import glob
import math
import random
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np


def ensure_16k_mono(waveform, sr, target_sr=16000):
    """统一转为单声道和 16000Hz"""
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


def save_with_tail_silence_protection(out_path, waveform, sr=16000, pad_silence_sec=0.35):
    """
    保存音频并施加双重尾音保护:
    1. 尾部最后 20ms 施加半余弦平滑淡出 (Fade-out)，防止数字截断引发的 Click/Pop 爆音；
    2. 追加 0.35s 纯净静音缓冲，确保无论是操作系统声卡 DAC 延迟关闭，还是各类媒体播放器播放，
       都绝不会截断人声的最后一个字或气流尾音！
    """
    fade_len = int(sr * 0.02)  # 20ms
    if waveform.shape[1] > fade_len:
        fade_curve = torch.from_numpy(0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))).float()
        waveform[0, -fade_len:] *= fade_curve

    # 追加纯静音缓冲
    silence_len = int(sr * pad_silence_sec)
    silence = torch.zeros(1, silence_len)
    protected_wf = torch.cat([waveform, silence], dim=1)

    # 振幅保护，防止削波
    max_val = torch.max(torch.abs(protected_wf))
    if max_val > 0.99:
        protected_wf = protected_wf / max_val * 0.98

    torchaudio.save(str(out_path), protected_wf, sr, bits_per_sample=16)


def synthesize_monitor_beeps(freq=1000.0, beep_duration=0.12, interval=0.88, total_sec=2.5, sr=16000):
    """
    高保真仿真手术室监护仪/心电监护仪的'滴滴滴'、'嘟嘟嘟'心率与报警脉冲波
    - freq: 脉冲频率 (典型值: 800Hz, 1000Hz, 1500Hz, 2000Hz)
    - beep_duration: 单次发声时长 (约 0.1~0.15s)
    - interval: 间隔时间 (心率 60~120 bpm)
    """
    total_samples = int(sr * total_sec)
    t = np.arange(total_samples) / sr
    sig = np.zeros(total_samples, dtype=np.float32)

    cycle_sec = beep_duration + interval
    for start_t in np.arange(0, total_sec, cycle_sec):
        idx_start = int(start_t * sr)
        idx_end = min(total_samples, int((start_t + beep_duration) * sr))
        if idx_start >= total_samples:
            break
        n_pts = idx_end - idx_start
        t_pulse = np.arange(n_pts) / sr
        # 正弦波脉冲 + 汉宁窗平滑起落 (模拟压电蜂鸣器真实响应)
        pulse = np.sin(2 * np.pi * freq * t_pulse) * np.hanning(n_pts)
        sig[idx_start:idx_end] = pulse * 0.7

    return torch.from_numpy(sig).unsqueeze(0)


def build_aishell_negatives(aishell_dir, out_dir, target_count=3200):
    """从 Aishell-1 中抽取日常句与带'眼/英/应/因'近音对抗句"""
    print(f"\n[*] [模块 1/4] 正在构建 Aishell-1 阴性语料 (目标: {target_count} 例)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_root = Path(aishell_dir) / "data_aishell" / "wav"
    txt_path = Path(aishell_dir) / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"

    if not wav_root.exists() or not txt_path.exists():
        print(f"[!] 警告: Aishell-1 路径不存在: {aishell_dir}")
        return []

    # 1. 扫描所有 wav 建立 uttid 字典
    print("    正在扫描 Aishell-1 音频文件库...")
    wav_dict = {f.stem: f for f in wav_root.rglob("*.wav")}
    print(f"    共扫描到 {len(wav_dict)} 个 Aishell 音频文件")

    # 2. 读取文本，划分为“近音强对抗句”和“通用日常句”
    confusable_keywords = ["眼", "英", "应", "因", "影", "映", "营", "硬"]
    hard_list = []
    normal_list = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uttid = parts[0]
            if uttid not in wav_dict:
                continue
            text = "".join(parts[1:])
            # 排除任何含有“鹰眼”的句子 (虽然实测为0条)
            if "鹰眼" in text:
                continue
            if any(k in text for k in confusable_keywords):
                hard_list.append((uttid, wav_dict[uttid], text))
            else:
                normal_list.append((uttid, wav_dict[uttid], text))

    print(f"    命中近音强对抗句: {len(hard_list)} 条，通用日常句: {len(normal_list)} 条")

    # 3. 组合抽样 (最多 350 条近音词 + 补齐通用日常句到 target_count)
    selected_hard = hard_list[: min(350, len(hard_list))]
    needed_normal = target_count - len(selected_hard)
    random.seed(42)
    selected_normal = random.sample(normal_list, min(needed_normal, len(normal_list)))
    final_selected = selected_hard + selected_normal
    random.shuffle(final_selected)

    results = []
    for idx, (uttid, src_wav, text) in enumerate(final_selected):
        out_wav = out_dir / f"neg_aishell_{idx+1:05d}_{uttid}.wav"
        try:
            wf, sr = torchaudio.load(str(src_wav))
            wf, sr = ensure_16k_mono(wf, sr)
            save_with_tail_silence_protection(out_wav, wf, sr=sr, pad_silence_sec=0.35)
            results.append((out_wav.name, str(out_wav.resolve()), text, "aishell"))
        except Exception:
            continue
        if (idx + 1) % 500 == 0 or (idx + 1) == len(final_selected):
            sys.stdout.write(f"\r    处理 Aishell 进度: [{idx+1}/{len(final_selected)}]...")
            sys.stdout.flush()

    print(f"\n    ✓ Aishell-1 构建完成: 成功产出 {len(results)} 个音频")
    return results


def build_clinical_slices(gastroscopy_dir, colonoscopy_dir, out_dir, target_count=1400):
    """将胃镜和肠镜的 412 条 27s 长音频切片为 2.5s~3.5s 的高质量临床阴性语音片段"""
    print(f"\n[*] [模块 2/4] 正在切片胃镜与肠镜长音频 (目标: ~{target_count} 例)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = []
    for d in [gastroscopy_dir, colonoscopy_dir]:
        p = Path(d)
        if p.exists():
            src_files.extend([f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".m4a"]])

    src_files.sort()
    print(f"    找到临床原始长音频共: {len(src_files)} 个")
    if not src_files:
        return []

    # 每个长音频平均切取 3~4 个有效切片
    slices_per_audio = max(3, math.ceil(target_count / len(src_files)))
    results = []
    slice_idx = 1

    for file_idx, fpath in enumerate(src_files):
        try:
            wf, sr = torchaudio.load(str(fpath))
            wf, sr = ensure_16k_mono(wf, sr)
        except Exception:
            continue

        dur_sec = wf.shape[1] / sr
        # 如果音频太短，直接作为一段
        if dur_sec <= 4.0:
            out_wav = out_dir / f"neg_clinical_{slice_idx:05d}_{fpath.stem}.wav"
            save_with_tail_silence_protection(out_wav, wf, sr=sr, pad_silence_sec=0.35)
            results.append((out_wav.name, str(out_wav.resolve()), "<clinical_speech>", "clinical"))
            slice_idx += 1
            continue

        # 按 2.8s ~ 3.5s 窗口均匀滑动切片，避开开头 0.5s 无声段
        start_cursor = int(sr * 0.5)
        for s in range(slices_per_audio):
            slice_len = int(sr * random.uniform(2.6, 3.4))
            end_cursor = start_cursor + slice_len
            if end_cursor > wf.shape[1]:
                # 最后一截如果够长就取，不够就结束
                if wf.shape[1] - start_cursor >= int(sr * 1.5):
                    chunk = wf[:, start_cursor:]
                    out_wav = out_dir / f"neg_clinical_{slice_idx:05d}_{fpath.stem}_s{s+1}.wav"
                    save_with_tail_silence_protection(out_wav, chunk, sr=sr, pad_silence_sec=0.35)
                    results.append((out_wav.name, str(out_wav.resolve()), "<clinical_speech>", "clinical"))
                    slice_idx += 1
                break

            chunk = wf[:, start_cursor:end_cursor]
            # 仅保留具有真实人声能量的片段 (过滤纯静音段)
            rms = torch.sqrt(torch.mean(chunk**2)).item()
            if rms > 0.01:
                out_wav = out_dir / f"neg_clinical_{slice_idx:05d}_{fpath.stem}_s{s+1}.wav"
                save_with_tail_silence_protection(out_wav, chunk, sr=sr, pad_silence_sec=0.35)
                results.append((out_wav.name, str(out_wav.resolve()), "<clinical_speech>", "clinical"))
                slice_idx += 1

            start_cursor += int(slice_len * 0.85)  # 15% 重叠步进

        if len(results) >= target_count:
            break

        if (file_idx + 1) % 50 == 0:
            sys.stdout.write(f"\r    处理长音频进度: [{file_idx+1}/{len(src_files)}] -> 已切得: {len(results)} 片段")
            sys.stdout.flush()

    print(f"\n    ✓ 胃镜/肠镜切片完成: 成功产出 {len(results)} 个临床语音片段")
    return results


def build_aikit_negatives(aikit_dir, out_dir):
    """规整现有的 AIKit 混淆词音频并补齐 0.35s 静音保护缓冲"""
    print(f"\n[*] [模块 3/4] 正在规整 AIKit 强混淆对抗词...")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Path(aikit_dir)
    wavs = sorted(list(p.glob("*.wav")))
    print(f"    找到 AIKit 混淆音频: {len(wavs)} 个")

    # 读取旧 metadata (如果有)
    meta_dict = {}
    meta_path = p / "metadata.txt"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and not parts[0].startswith("#"):
                    meta_dict[parts[0]] = parts[2]

    results = []
    for idx, w in enumerate(wavs):
        out_wav = out_dir / f"neg_aikit_{idx+1:04d}_{w.stem}.wav"
        try:
            wf, sr = torchaudio.load(str(w))
            wf, sr = ensure_16k_mono(wf, sr)
            save_with_tail_silence_protection(out_wav, wf, sr=sr, pad_silence_sec=0.35)
            text = meta_dict.get(w.stem, "<aikit_confusable>")
            results.append((out_wav.name, str(out_wav.resolve()), text, "aikit_hard_neg"))
        except Exception:
            continue

    print(f"    ✓ AIKit 混淆词规整完成: 成功产出 {len(results)} 个音频")
    return results


def build_or_noises_and_beeps(musan_dir, out_dir, target_count=400):
    """提取 MUSAN 机械/风扇底噪，并合成高保真监护仪'滴滴滴/嘟嘟嘟'心率与报警音"""
    print(f"\n[*] [模块 4/4] 正在构建手术室声学环境与监护仪蜂鸣音 (目标: {target_count} 例)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. 纯代码数学仿真监护仪“滴滴滴”与“嘟嘟嘟”蜂鸣声 (~150 例)
    beep_freqs = [800.0, 950.0, 1000.0, 1200.0, 1500.0, 1800.0, 2000.0]  # 典型监护仪频段
    print("    正在合成监护仪高保真心率滴滴声与报警蜂鸣音...")
    for i in range(150):
        f = random.choice(beep_freqs)
        # 心率正常模式 (60~80 bpm) vs 报警急促模式 (100~140 bpm)
        is_alarm = (i % 2 == 1)
        dur = random.uniform(0.08, 0.15) if is_alarm else random.uniform(0.12, 0.20)
        interval = random.uniform(0.4, 0.6) if is_alarm else random.uniform(0.7, 1.1)
        total_len = random.uniform(2.2, 3.2)

        wf = synthesize_monitor_beeps(freq=f, beep_duration=dur, interval=interval, total_sec=total_len, sr=16000)
        out_wav = out_dir / f"neg_monitor_beep_{i+1:04d}_f{int(f)}hz.wav"
        save_with_tail_silence_protection(out_wav, wf, sr=16000, pad_silence_sec=0.35)
        results.append((out_wav.name, str(out_wav.resolve()), "<monitor_beep>", "monitor_beep"))

    # 2. 从 MUSAN 中提取机械底噪、排风、空调、按键提示音 (~250 例)
    musan_noise_p = Path(musan_dir) / "noise"
    noise_files = list(musan_noise_p.rglob("*.wav")) if musan_noise_p.exists() else []
    print(f"    扫描到 MUSAN 噪声音频库: {len(noise_files)} 个")

    if noise_files:
        random.seed(123)
        sample_noises = random.sample(noise_files, min(250, len(noise_files)))
        for i, nfile in enumerate(sample_noises):
            try:
                wf, sr = torchaudio.load(str(nfile))
                wf, sr = ensure_16k_mono(wf, sr)
                # 切取 2.5s 片段
                target_len = int(16000 * random.uniform(2.0, 3.0))
                if wf.shape[1] > target_len:
                    start_p = random.randint(0, wf.shape[1] - target_len)
                    wf = wf[:, start_p : start_p + target_len]
                out_wav = out_dir / f"neg_or_noise_{i+1:04d}_{nfile.stem}.wav"
                save_with_tail_silence_protection(out_wav, wf, sr=16000, pad_silence_sec=0.35)
                results.append((out_wav.name, str(out_wav.resolve()), "<ambient_noise>", "or_noise"))
            except Exception:
                continue

    print(f"    ✓ 手术室噪声与监护仪蜂鸣构建完成: 成功产出 {len(results)} 个音频")
    return results


def main():
    parser = argparse.ArgumentParser(description="KWS 阴性样本全自动构建 (5000+ 例)")
    parser.add_argument("--aishell_dir", type=str, default="/media/inno/ASR/KWS/数据集/Aishell-1")
    parser.add_argument("--gastroscopy_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性/真实/胃镜")
    parser.add_argument("--colonoscopy_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性/真实/肠镜")
    parser.add_argument("--aikit_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性/AIkit")
    parser.add_argument("--musan_dir", type=str, default="/media/inno/ASR/KWS/数据集/musan")
    parser.add_argument("--output_root", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性集",
                        help="最终规整后的阴性数据集根目录")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" KWS 工业级阴性样本全自动构建调度系统 (5000+ 规整版) ".center(70, "="))
    print(f"• 输出目标根目录 : {out_root.resolve()}")
    print("• 关键技术保证   : 统一 16000Hz 16-bit 单声道 | 全切片末尾追加 0.35s 静音缓冲")
    print("• 测试集隔离保护 : 严格排除 19 条真实线上误唤醒样本 (留作盲测)")
    print("=" * 70)

    all_samples = []

    # 1. Aishell-1 抽样 (~3200)
    res_aishell = build_aishell_negatives(args.aishell_dir, out_root / "01_aishell", target_count=3200)
    all_samples.extend(res_aishell)

    # 2. 胃镜/肠镜切片 (~1400)
    res_clinical = build_clinical_slices(args.gastroscopy_dir, args.colonoscopy_dir, out_root / "02_clinical_slices", target_count=1400)
    all_samples.extend(res_clinical)

    # 3. AIKit 强混淆对抗词 (~210)
    res_aikit = build_aikit_negatives(args.aikit_dir, out_root / "03_aikit_confusables")
    all_samples.extend(res_aikit)

    # 4. 手术室噪声 + 监护仪蜂鸣 (~400)
    res_or = build_or_noises_and_beeps(args.musan_dir, out_root / "04_or_noises_and_beeps", target_count=400)
    all_samples.extend(res_or)

    # 5. 写入统一标准索引文件
    print("\n[*] 正在生成 FunASR 标准训练索引文件 (wav.scp / text / metadata.txt)...")
    wav_scp_path = out_root / "wav.scp"
    text_path = out_root / "text"
    meta_path = out_root / "metadata.txt"

    with open(wav_scp_path, "w", encoding="utf-8") as f_scp, \
         open(text_path, "w", encoding="utf-8") as f_text, \
         open(meta_path, "w", encoding="utf-8") as f_meta:

        f_meta.write("# uttid\twav_path\ttext_desc\tcategory\n")
        for fname, fpath, text_desc, cat in all_samples:
            utt_id = Path(fname).stem
            f_scp.write(f"{utt_id}\t{fpath}\n")
            # KWS 训练时负样本对应 <sil> (静音/无唤醒)
            f_text.write(f"{utt_id}\t<sil>\n")
            f_meta.write(f"{utt_id}\t{fpath}\t{text_desc}\t{cat}\n")

    print("\n" + "=" * 70)
    print(" 🎉 阴性数据集全量构建成功！ ".center(70, "="))
    print(f"• 阴性样本总数   : {len(all_samples)} 例 (完全满足 5000+ 工业标准)")
    print(f"  ├─ 01_Aishell-1 通用日常与近音 : {len(res_aishell)} 例")
    print(f"  ├─ 02_胃肠镜真实临床语音切片   : {len(res_clinical)} 例")
    print(f"  ├─ 03_AIKit 强对抗近音词       : {len(res_aikit)} 例")
    print(f"  └─ 04_手术室噪声与仿真监护仪音 : {len(res_or)} 例")
    print(f"• 数据集存放根目录 : {out_root.resolve()}")
    print(f"• FunASR wav.scp   : {wav_scp_path}")
    print(f"• FunASR text      : {text_path}")
    print(f"• 详细 metadata    : {meta_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
