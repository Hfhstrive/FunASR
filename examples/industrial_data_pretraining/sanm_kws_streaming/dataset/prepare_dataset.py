#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唤醒词 (KWS) 微调训练数据集生成与分层平衡脚本 (v3 规整版)
运行环境: torch2.7.1 (/home/inno/anaconda3/envs/torch2.7.1/bin/python)

针对输入:
  • 阳性数据源 1: /media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实_增强 (1611例, 14个发音人)
  • 阳性数据源 2: /media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/AIKit     (201例, 多发音人)
  • 阴性数据源  : /media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性集         (5210例, 4大子类)
输出目标:
  • /media/inno/ASR/KWS/TrainData/kws_v3 (纯英文路径，彻底规避 Hydra 中文编码限制)

核心机制:
1. 零拷贝符号链接 (Zero-Copy Symlinks):
   所有音频已在预处理阶段规整为标准 16000Hz 16-bit 单声道 WAV，
   脚本在 TrainData/kws_v3/wavs/ 下极速创建软链接，不占用额外磁盘空间，秒级构建完成！
2. 严格分层均匀划分验证集 (Stratified Split):
   - 阳性：按每个说话人子组分别按 15% 独立抽取放入 val，确保验证集全量覆盖所有真实与合成发音人；
   - 阴性：按 Aishell、临床切片、AIKit 混淆词、手术室噪声 4 个子类分别按 15% 独立抽取放入 val；
3. 输出完整的 FunASR 训练所需全套索引:
   train.jsonl, val.jsonl, train_wav.scp, val_wav.scp, train_text.txt, val_text.txt
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import torchaudio

TARGET_SR = 16000
KEYWORD = "鹰眼鹰眼"
VAL_RATIO = 0.15
SEED = 42


def get_audio_duration(file_path):
    """极速获取音频时长与采样率，无需解压波形"""
    try:
        info = torchaudio.info(str(file_path))
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"[!] 无法读取音频信息: {file_path}, 错误: {e}")
        return None


def stratified_split(records_by_group, val_ratio=VAL_RATIO):
    """
    分层均匀划分：对每个分组独立按 val_ratio 抽取放入验证集，其余放入训练集
    """
    train_records = []
    val_records = []
    group_stats = {}

    for group_name, items in records_by_group.items():
        shuffled = list(items)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0

        v_items = shuffled[:n_val]
        t_items = shuffled[n_val:]

        val_records.extend(v_items)
        train_records.extend(t_items)

        group_stats[group_name] = {
            "total": n_total,
            "train": len(t_items),
            "val": len(v_items)
        }

    return train_records, val_records, group_stats


def main():
    parser = argparse.ArgumentParser(description="FunASR KWS 训练数据集全自动分层构建")
    parser.add_argument("--pos_real_dir", type=str,
                        default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/真实_增强",
                        help="真实阳性增强样本目录")
    parser.add_argument("--pos_aikit_dir", type=str,
                        default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/AIKit",
                        help="AIKit 合成阳性样本目录")
    parser.add_argument("--neg_root_dir", type=str,
                        default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阴性集",
                        help="规整后的阴性数据集根目录")
    parser.add_argument("--out_dir", type=str,
                        default="/media/inno/ASR/KWS/TrainData/kws_v3",
                        help="输出训练数据根目录 (必须为纯英文路径)")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO,
                        help="验证集划分比例 (默认 0.15)")
    args = parser.parse_args()

    random.seed(SEED)

    out_base_dir = Path(args.out_dir)
    out_symlink_dir = out_base_dir / "wavs"
    out_symlink_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" FunASR 唤醒词 (KWS) 训练数据集构建与分层平衡调度系统 ".center(70, "="))
    print(f"• 阳性数据源 1 (真实增强) : {args.pos_real_dir}")
    print(f"• 阳性数据源 2 (AIKit合成) : {args.pos_aikit_dir}")
    print(f"• 阴性数据源 (4维阴性库)  : {args.neg_root_dir}")
    print(f"• 输出目标目录 (纯英文)   : {out_base_dir.resolve()}")
    print(f"• 验证集抽取比例         : {args.val_ratio * 100:.1f}%")
    print("=" * 70 + "\n")

    # ==================== 1. 扫描并处理阳性样本 ====================
    print("[*] 正在扫描并索引【阳性样本】...")
    pos_records_by_group = {}
    pos_total_dur = 0.0
    pos_idx = 0

    # 1.1 扫描真实增强样本 (按各个发音人分组)
    real_p = Path(args.pos_real_dir)
    if real_p.exists():
        for spk_dir in sorted([x for x in real_p.iterdir() if x.is_dir()]):
            spk_name = f"real_{spk_dir.name}"
            pos_records_by_group[spk_name] = []
            wav_files = sorted(list(spk_dir.glob("*.wav")))
            for w in wav_files:
                dur = get_audio_duration(w)
                if dur is None or dur <= 0.1:
                    continue
                pos_idx += 1
                utt_id = f"pos_{pos_idx:05d}_{spk_dir.name}_{w.stem}"
                # 创建软链接 (纯英文文件名)
                symlink_path = out_symlink_dir / f"{utt_id}.wav"
                if not symlink_path.exists():
                    symlink_path.symlink_to(w.resolve())

                pos_records_by_group[spk_name].append((utt_id, str(symlink_path.resolve()), KEYWORD, dur, spk_name))
                pos_total_dur += dur

    # 1.2 扫描 AIKit 合成样本 (按发音人分组)
    aikit_p = Path(args.pos_aikit_dir)
    if aikit_p.exists():
        wav_files = sorted(list(aikit_p.glob("*.wav")))
        for w in wav_files:
            dur = get_audio_duration(w)
            if dur is None or dur <= 0.1:
                continue
            # 提取发音人名称 (yy_pos_aikit_0001_xiaoyan_s42_p48)
            parts = w.stem.split("_")
            vcn = parts[4] if len(parts) >= 5 else "aikit_synth"
            group_name = f"aikit_{vcn}"
            pos_records_by_group.setdefault(group_name, [])

            pos_idx += 1
            utt_id = f"pos_{pos_idx:05d}_{vcn}_{w.stem}"
            symlink_path = out_symlink_dir / f"{utt_id}.wav"
            if not symlink_path.exists():
                symlink_path.symlink_to(w.resolve())

            pos_records_by_group[group_name].append((utt_id, str(symlink_path.resolve()), KEYWORD, dur, group_name))
            pos_total_dur += dur

    print(f"[+] 阳性样本索引完毕: 共 {pos_idx} 例，覆盖 {len(pos_records_by_group)} 个细分发音组，总时长: {pos_total_dur/60:.2f} 分钟\n")

    # ==================== 2. 扫描并处理阴性样本 ====================
    print("[*] 正在扫描并索引【阴性样本】(4大子类别)...")
    neg_records_by_category = {}
    neg_total_dur = 0.0
    neg_idx = 0

    neg_p = Path(args.neg_root_dir)
    if not neg_p.exists():
        print(f"[!] 错误: 阴性根目录不存在: {args.neg_root_dir}")
        sys.exit(1)

    neg_categories = sorted([x for x in neg_p.iterdir() if x.is_dir()])
    for cat_dir in neg_categories:
        cat_name = cat_dir.name
        neg_records_by_category[cat_name] = []
        wav_files = sorted(list(cat_dir.glob("*.wav")))
        for w in wav_files:
            dur = get_audio_duration(w)
            if dur is None or dur <= 0.1:
                continue
            neg_idx += 1
            utt_id = f"neg_{neg_idx:05d}_{cat_name}_{w.stem}"
            symlink_path = out_symlink_dir / f"{utt_id}.wav"
            if not symlink_path.exists():
                symlink_path.symlink_to(w.resolve())

            neg_records_by_category[cat_name].append((utt_id, str(symlink_path.resolve()), "<sil>", dur, cat_name))
            neg_total_dur += dur

    print(f"[+] 阴性样本索引完毕: 共 {neg_idx} 例，覆盖 {len(neg_records_by_category)} 大类，总时长: {neg_total_dur/60:.2f} 分钟\n")

    # ==================== 3. 严格分层均匀划分 ====================
    print("[*] 正在执行严格分层均分切分 (阳性按发音人均分、阴性按4大子类均分)...")
    train_pos, val_pos, pos_stats = stratified_split(pos_records_by_group, val_ratio=args.val_ratio)
    train_neg, val_neg, neg_stats = stratified_split(neg_records_by_category, val_ratio=args.val_ratio)

    train_all = train_pos + train_neg
    val_all = val_pos + val_neg

    random.shuffle(train_all)
    random.shuffle(val_all)

    # ==================== 4. 打印统计报告 ====================
    print("=" * 70)
    print(" 数据集分层划分统计报告 ".center(70, "="))
    print("\n【阳性细分组分布】:")
    for grp, st in sorted(pos_stats.items()):
        print(f"  • {grp:<18}: 总数 {st['total']:<4} | 训练集 {st['train']:<4} | 验证集 {st['val']}")

    print("\n【阴性子类别分布】:")
    for cat, st in sorted(neg_stats.items()):
        print(f"  • {cat:<26}: 总数 {st['total']:<4} | 训练集 {st['train']:<4} | 验证集 {st['val']}")

    train_pos_dur = sum(x[3] for x in train_pos) / 60
    train_neg_dur = sum(x[3] for x in train_neg) / 60
    val_pos_dur = sum(x[3] for x in val_pos) / 60
    val_neg_dur = sum(x[3] for x in val_neg) / 60

    print("\n【总览汇总与黄金比例】:")
    print(f"  • 训练集总计: {len(train_all)} 条 | 阳性: {len(train_pos)} 条 ({train_pos_dur:.1f}m) | 阴性: {len(train_neg)} 条 ({train_neg_dur:.1f}m)")
    print(f"  • 验证集总计: {len(val_all)} 条   | 阳性: {len(val_pos)} 条 ({val_pos_dur:.1f}m)   | 阴性: {len(val_neg)} 条 ({val_neg_dur:.1f}m)")
    print(f"  • 条数比例 (阳性 : 阴性) = 1 : {(neg_idx / pos_idx):.2f}")
    print(f"  • 时长比例 (阳性 : 阴性) = 1 : {(neg_total_dur / pos_total_dur):.2f}")
    print("=" * 70 + "\n")

    # ==================== 5. 写入索引文件 (scp, text, jsonl) ====================
    for split_name, dataset in [("train", train_all), ("val", val_all)]:
        wav_scp_path = out_base_dir / f"{split_name}_wav.scp"
        text_txt_path = out_base_dir / f"{split_name}_text.txt"
        jsonl_path = out_base_dir / f"{split_name}.jsonl"

        with open(wav_scp_path, "w", encoding="utf-8") as fw, \
             open(text_txt_path, "w", encoding="utf-8") as ft, \
             open(jsonl_path, "w", encoding="utf-8") as fj:
            for utt_id, wav_path, text, dur, _ in dataset:
                fw.write(f"{utt_id}\t{wav_path}\n")
                ft.write(f"{utt_id}\t{text}\n")

                # 生成标准 jsonl (用于 FunASR AudioDataset)
                source_len = int(dur * 100)
                target_len = 0 if text in ["<sil>", "!sil", ""] else len(text)
                record = {
                    "key": utt_id,
                    "source": wav_path,
                    "source_len": source_len,
                    "target": text,
                    "target_len": target_len
                }
                fj.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[+] 已生成索引文件: {wav_scp_path}")
        print(f"[+] 已生成文本文件: {text_txt_path}")
        print(f"[+] 已生成标准数据: {jsonl_path}")

    print(f"\n🎉 恭喜！KWS 训练与验证数据集已全部成功构建就绪！存放于: {out_base_dir.resolve()}\n")


if __name__ == "__main__":
    main()
