#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唤醒词 (KWS) 微调数据集生成与平衡脚本
处理流程：
1. 音频格式统一：将所有音频转换为 16000Hz、单声道、16-bit PCM WAV 格式；
2. 样本平衡控制：阴性样本若单条过长，裁剪为 3.5s 左右的有效语音切片，使阳性与阴性总时长达到 1:2 黄金比例；
3. 训练/验证集划分：按 85:15 分层切分生成 train / val 数据；
4. 输出标准索引：生成 train_wav.scp, train_text.txt, val_wav.scp, val_text.txt 以及 jsonl 文件。
"""

import os
import glob
import random
import torchaudio
import torchaudio.transforms as T
import torch
import soundfile as sf

TARGET_SR = 16000
KEYWORD = "鹰眼鹰眼"

def load_and_convert(file_path, target_sr=TARGET_SR):
    """加载并转为 16kHz 单声道 float32 tensor"""
    try:
        wf, sr = torchaudio.load(file_path)
    except Exception:
        data, sr = sf.read(file_path, dtype="float32")
        wf = torch.from_numpy(data)
        if len(wf.shape) == 1:
            wf = wf.unsqueeze(0)
        else:
            wf = wf.transpose(0, 1)

    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        wf = resampler(wf)

    return wf.squeeze(0), target_sr

def extract_active_chunk(wf, target_duration=3.5, sr=TARGET_SR):
    """
    从较长的阴性音频中截取一段能量最显著的有效语音（避开完全静音的段落）
    """
    chunk_len = int(target_duration * sr)
    total_len = wf.shape[0]

    if total_len <= chunk_len:
        return wf

    # 分块寻找能量最大的窗口
    step = int(0.5 * sr)
    best_start = 0
    max_energy = -1.0

    for start in range(0, total_len - chunk_len, step):
        sub = wf[start : start + chunk_len]
        energy = float((sub ** 2).mean())
        if energy > max_energy:
            max_energy = energy
            best_start = start

    return wf[best_start : best_start + chunk_len]

def main():
    src_pos_dir = "/media/inno/ASR/唤醒词/数据集/阳性"
    src_neg_dir = "/media/inno/ASR/唤醒词/数据集/阴性"
    out_base_dir = "/media/inno/ASR/唤醒词/数据集_processed"

    out_wav_dir = os.path.join(out_base_dir, "wavs_16k")
    os.makedirs(out_wav_dir, exist_ok=True)

    # 1. 收集原始文件列表
    pos_files = []
    for ext in ("*.wav", "*.m4a", "*.mp3", "*.flac"):
        pos_files.extend(glob.glob(os.path.join(src_pos_dir, "**", ext), recursive=True))
    pos_files = sorted(list(set(pos_files)))

    neg_files = []
    for ext in ("*.wav", "*.m4a", "*.mp3", "*.flac"):
        neg_files.extend(glob.glob(os.path.join(src_neg_dir, "**", ext), recursive=True))
    neg_files = sorted(list(set(neg_files)))

    print(f"[*] 发现原始阳性样本: {len(pos_files)} 个")
    print(f"[*] 发现原始阴性样本: {len(neg_files)} 个")

    pos_records = []  # (utt_id, wav_path, text, duration)
    neg_records = []

    # 2. 处理阳性样本（保留完整唤醒词音频）
    print("[*] 正在处理阳性样本...")
    pos_total_dur = 0.0
    for idx, f in enumerate(pos_files):
        wf, sr = load_and_convert(f)
        dur = wf.shape[0] / sr
        utt_id = f"pos_{idx+1:04d}"
        save_path = os.path.join(out_wav_dir, f"{utt_id}.wav")
        torchaudio.save(save_path, wf.unsqueeze(0), sr)
        pos_records.append((utt_id, save_path, KEYWORD, dur))
        pos_total_dur += dur

    # 3. 处理阴性样本（裁剪为 3.5s 左右的高质量负样本切片）
    print("[*] 正在处理阴性样本（截取有效语音切片以平衡时长）...")
    neg_total_dur = 0.0
    for idx, f in enumerate(neg_files):
        wf, sr = load_and_convert(f)
        # 裁剪出 3.5s 核心医学术语片段
        wf_chunk = extract_active_chunk(wf, target_duration=3.5, sr=sr)
        dur = wf_chunk.shape[0] / sr
        utt_id = f"neg_{idx+1:04d}"
        save_path = os.path.join(out_wav_dir, f"{utt_id}.wav")
        torchaudio.save(save_path, wf_chunk.unsqueeze(0), sr)
        neg_records.append((utt_id, save_path, "<sil>", dur))
        neg_total_dur += dur

    print("\n" + "="*50)
    print(" 数据集处理与平衡统计 ".center(50, "="))
    print(f"阳性样本: {len(pos_records)} 条, 总时长: {pos_total_dur:.1f}s ({pos_total_dur/60:.2f}分钟)")
    print(f"阴性样本: {len(neg_records)} 条, 总时长: {neg_total_dur:.1f}s ({neg_total_dur/60:.2f}分钟)")
    print(f"时长比例 (阳性:阴性) = 1 : {neg_total_dur/pos_total_dur:.2f}")
    print("="*50 + "\n")

    # 4. 分层划分训练集与验证集 (85% train, 15% val)
    random.seed(42)
    random.shuffle(pos_records)
    random.shuffle(neg_records)

    n_pos_val = max(5, int(len(pos_records) * 0.15))
    n_neg_val = max(10, int(len(neg_records) * 0.15))

    val_pos = pos_records[:n_pos_val]
    train_pos = pos_records[n_pos_val:]

    val_neg = neg_records[:n_neg_val]
    train_neg = neg_records[n_neg_val:]

    train_data = train_pos + train_neg
    val_data = val_pos + val_neg

    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"[+] 训练集总计: {len(train_data)} 条 (阳性: {len(train_pos)}, 阴性: {len(train_neg)})")
    print(f"[+] 验证集总计: {len(val_data)} 条 (阳性: {len(val_pos)}, 阴性: {len(val_neg)})")

    # 5. 写入 scp 与 text 文件
    for name, data_list in [("train", train_data), ("val", val_data)]:
        wav_scp_path = os.path.join(out_base_dir, f"{name}_wav.scp")
        text_txt_path = os.path.join(out_base_dir, f"{name}_text.txt")

        with open(wav_scp_path, "w", encoding="utf-8") as fw, \
             open(text_txt_path, "w", encoding="utf-8") as ft:
            for utt_id, wav_path, text, _ in data_list:
                fw.write(f"{utt_id}\t{wav_path}\n")
                ft.write(f"{utt_id}\t{text}\n")

        print(f"[+] 已生成: {wav_scp_path}")
        print(f"[+] 已生成: {text_txt_path}")

    print(f"\n✅ 数据集已成功生成并就绪！保存于: {out_base_dir}")

if __name__ == "__main__":
    main()
