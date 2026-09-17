#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将微调后的 KWS 唤醒词模型导出为 ONNX 格式。
使用 AutoModel 实例方法 export()，避免顶层 export() 的传参问题。
"""

import os
from funasr import AutoModel


def main():
    # ========== 配置区 ==========
    MODEL_PATH = "/media/inno/work_dirs/ASR/kws_yingyan"
    CHECKPOINT = "/media/inno/work_dirs/ASR/kws_yingyan/model.pt.best"
    OUTPUT_DIR = "/media/inno/work_dirs/ASR/kws_yingyan/onnx"

    OPSET_VERSION = 14
    QUANTIZE = False
    # ===========================

    print(f"[*] 正在加载模型: {MODEL_PATH}")
    print(f"[*] 使用权重: {CHECKPOINT}")

    model = AutoModel(
        model=MODEL_PATH,
        init_param=CHECKPOINT,
        device="cpu",
        disable_update=True
    )

    print(f"[*] 开始导出 ONNX (opset={OPSET_VERSION}, quantize={QUANTIZE})...")

    # 使用实例方法，而非 funasr.utils.export_utils.export
    res = model.export(
        type="onnx",
        output_dir=OUTPUT_DIR,
        opset_version=OPSET_VERSION,
        quantize=QUANTIZE
    )

    print(f"[+] 导出完成，结果: {res}")


if __name__ == "__main__":
    main()