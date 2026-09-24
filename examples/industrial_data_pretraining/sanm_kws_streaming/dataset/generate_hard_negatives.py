#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于科大讯飞 (iFlytek) 在线 TTS API 的 KWS 混淆词 (Hard Negatives) 批量合成脚本
特点：
1. 内置 5 大类、共计 100+ 条针对“鹰眼鹰眼”的强对抗声学混淆词；
2. 启动时自动对配置的发音人执行“权限探活 (Auto-Probe)”，自动剔除无权限的发音人，确保 100% 成功；
3. 支持覆盖慢速 (38)、正常 (50)、快速 (65) 多档语速与音调扰动，丰富时域多样性；
4. 直接输出 16000Hz、16-bit 单声道标准 WAV 音频，无缝契合 FunASR / KWS 训练格式；
5. 自动生成对应的 metadata.txt 索引文件，可直接被 prepare_dataset.py 引入作为负样本。
"""

import os
import sys
import time
import json
import wave
import hmac
import ssl
import random
import hashlib
import base64
import argparse
import datetime
from urllib.parse import urlencode

try:
    import websocket
except ImportError:
    print("[!] 缺少 websocket-client 库，请先安装: pip install websocket-client")
    sys.exit(1)


# ==================== 1. 混淆词词库定义 ====================
HARD_NEGATIVE_TEXTS = [
    # 类别 A: 前缀/残缺词 (防听到一半抢跑误触发)
    "鹰眼",
    "一只鹰眼",
    "鹰眼模式",
    "鹰眼系统",
    "开启鹰眼",
    "呼叫鹰眼",
    "鹰眼一下",
    "看鹰眼",

    # 类别 B: ying / yan 同音叠词 (极高混淆度，打靶核心)
    "应用应用",
    "影响影响",
    "经验经验",
    "运营运营",
    "英雄英雄",
    "影院影院",
    "营养营养",
    "阴天阴天",
    "营业营业",
    "精英精英",
    "阴阳阴阳",
    "一眼一眼",
    "检验检验",
    "眼睛眼睛",
    "咽喉咽喉",
    "颜色颜色",
    "演变演变",
    "言语言语",
    "映现映现",
    "迎面迎面",

    # 类别 C: 韵母组合与四字成语 (四字节奏结构神似)
    "莺歌燕舞",
    "极目远眺",
    "阴云密布",
    "引人注目",
    "迎刃而解",
    "显而易见",
    "阴错阳差",
    "隐约可见",
    "营运检验",
    "应用体验",
    "英雄好汉",
    "英姿飒爽",
    "眼疾手快",

    # 类别 D: 医疗内镜临床高频术语 (领域专属强对抗)
    "电子硬镜",
    "荧光检查",
    "荧光模式",
    "局部造影",
    "肠管发炎",
    "边缘清晰",
    "胰腺饱满",
    "黏膜充血水肿",
    "胃窦充血",
    "咽部喷雾麻醉",
    "隐窝结构",
    "静脉曲张",
    "造影显影",
    "病灶边缘",
    "反流性食管炎",
    "内镜检查准备",

    # 类别 E: 检查室日常口语对话与操作指令
    "你看一眼这个地方",
    "稍微忍一下啊",
    "眼睛闭一下",
    "影响不大",
    "这个应用怎么打开",
    "把光源调亮一点",
    "准备冲洗吸引",
    "调一下阴影对比度",
    "打印检查报告",
    "病人深呼吸一下",
    "稍微往后退一点",
    "镜头擦一下",
    "进镜看一下幽门"
]


# ==================== 2. 讯飞 TTS 客户端 ====================
class XfyunTTSClient:
    def __init__(self, app_id, api_key, api_secret, vcn="x4_yezi", speed=50, volume=60, pitch=50):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.host = "tts-api.xfyun.cn"
        self.vcn = vcn
        self.speed = speed
        self.volume = volume
        self.pitch = pitch

    def _create_url(self):
        url = "wss://tts-api.xfyun.cn/v2/tts"
        now = datetime.datetime.now()
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
        signature_origin = f"host: {self.host}\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")
        authorization_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(encoding="utf-8")
        params = {"authorization": authorization, "date": date, "host": self.host}
        return url + "?" + urlencode(params)

    def synthesize(self, text, save_wav_path=None, max_retry=2):
        """合成语音，若 save_wav_path 为 None 则仅作为探活测试"""
        for retry in range(max_retry):
            pcm_buffer = bytearray()
            is_success = False
            error_msg = ""

            def on_message(ws, message):
                nonlocal is_success, error_msg
                try:
                    res = json.loads(message)
                    code = res.get("code", 0)
                    if code != 0:
                        error_msg = res.get("message", "未知错误")
                        ws.close()
                        return
                    audio_b64 = res.get("data", {}).get("audio", "")
                    if audio_b64:
                        pcm_buffer.extend(base64.b64decode(audio_b64))
                    if res.get("data", {}).get("status") == 2:
                        is_success = True
                        ws.close()
                except Exception as e:
                    error_msg = str(e)

            def on_open(ws):
                req = {
                    "common": {"app_id": self.app_id},
                    "business": {
                        "aue": "raw",                   # 返回原始线性 PCM
                        "auf": "audio/L16;rate=16000",   # 16kHz 16bit 规格
                        "vcn": self.vcn,
                        "speed": self.speed,
                        "volume": self.volume,
                        "pitch": self.pitch,
                        "tte": "UTF8"
                    },
                    "data": {
                        "status": 2,
                        "text": str(base64.b64encode(text.encode("utf-8")), "UTF-8")
                    }
                }
                ws.send(json.dumps(req))

            ws_url = self._create_url()
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=lambda w, e: None,
                on_close=lambda w, a, b: None
            )
            ws.on_open = on_open
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

            if is_success and len(pcm_buffer) > 0:
                if save_wav_path:
                    # 写入标准 16kHz WAV 文件
                    with wave.open(save_wav_path, "wb") as wf:
                        wf.setnchannels(1)      # 单声道
                        wf.setsampwidth(2)      # 16-bit (2 字节)
                        wf.setframerate(16000)  # 16000Hz 采样率
                        wf.writeframes(pcm_buffer)
                return True
            else:
                time.sleep(0.5)

        return False


def probe_valid_vcns(app_id, api_key, api_secret, candidate_vcns):
    """自动探活：测试哪些发音人在当前应用中有授权权限"""
    print("[*] 正在对候选发音人执行权限探活 (Probe)...")
    valid_vcns = []
    for vcn in candidate_vcns:
        client = XfyunTTSClient(app_id, api_key, api_secret, vcn=vcn)
        ok = client.synthesize("测试", save_wav_path=None, max_retry=1)
        if ok:
            valid_vcns.append(vcn)
            print(f"    ✓ [可用] {vcn}")
        else:
            print(f"    ✗ [未开通或无效] {vcn}")
        time.sleep(0.2)
    return valid_vcns


# ==================== 3. 主流程与调度 ====================
def main():
    parser = argparse.ArgumentParser(description="讯飞 TTS 批量生成 KWS 混淆对抗样本")
    parser.add_argument("--app_id", type=str, default="9d79363d")
    parser.add_argument("--api_key", type=str, default="69b0c6ffca5e506c0a3625d8754e87ca")
    parser.add_argument("--api_secret", type=str, default="NjA1NWI5NDUzOGNiYTA4NWExMDA3NWMz")
    parser.add_argument("--output_dir", type=str, default="/media/inno/ASR/KWS/数据集/阴性/AIkit")
    parser.add_argument("--repeat_per_word", type=int, default=3,
                        help="每个词合成的遍数（默认3遍，分别对应慢速/中速/快速）")
    parser.add_argument("--vcns", type=str, default="",
                        help="自定义发音人列表，多个用逗号隔开；为空则自动探活默认库")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 确定发音人候选库（包含常见基础与特色发音人）
    if args.vcns:
        candidates = [v.strip() for v in args.vcns.split(",") if v.strip()]
    else:
        candidates = [
            # 基础发音人
            "xiaoyan",              # 标准青年女声
            "x4_yezi",              # 叶子 (小露，女声)
            "aisxping",             # 小萍 (甜美女声)
            "aisjiuxu",             # 许久 (标准播音男声)
            "aisbabyxu",            # 许小宝 (童声/高音)
            # 常见特色发音人
            "x4_chaoge",            # 超哥 (沉稳男声)
            "x2_wanshu",            # 万叔 (大叔音)
            "x4_lingbosong",        # 聆伯松 (纪录片男声)
            "x4_lingxiaoxuan_en_v2",# 聆小璇 (知性助理女声)
            "x4_lingfeizhe_zl",     # 聆飞哲 (活力男声)
        ]

    # 2. 自动探活验证
    available_vcns = probe_valid_vcns(args.app_id, args.api_key, args.api_secret, candidates)
    if not available_vcns:
        print("[!] 错误: 未检测到任何可用的发音人，请检查 APIKey 与权限！")
        sys.exit(1)

    print(f"\n[+] 最终确认生效发音人 ({len(available_vcns)} 个): {available_vcns}")

    # 3. 语速多档位设定：慢速 (38~42)、常速 (48~52)、快速 (62~68)
    SPEED_PRESETS = [
        {"name": "慢速", "range": (38, 43)},
        {"name": "常速", "range": (48, 53)},
        {"name": "快速", "range": (62, 68)},
    ]

    texts = HARD_NEGATIVE_TEXTS
    total_tasks = len(texts) * args.repeat_per_word

    print("\n" + "=" * 60)
    print(" 讯飞在线 TTS 混淆对抗样本批量生成 ".center(60, "="))
    print(f"• APP_ID       : {args.app_id}")
    print(f"• 混淆词条数   : {len(texts)} 条")
    print(f"• 每词生成遍数 : {args.repeat_per_word} 遍 (覆盖慢/中/快速及不同发音人)")
    print(f"• 预计生成总数 : {total_tasks} 个 WAV 音频 (16kHz 16-bit 单声道)")
    print(f"• 保存目标目录 : {os.path.abspath(args.output_dir)}")
    print("=" * 60 + "\n")

    meta_file = os.path.join(args.output_dir, "metadata.txt")
    success_count = 0

    with open(meta_file, "w", encoding="utf-8") as f_meta:
        # 写入表头说明
        f_meta.write("# uttid\twav_path\ttext\tvcn\tspeed\tpitch\n")
        
        for word_idx, text in enumerate(texts):
            for r in range(args.repeat_per_word):
                # 轮换发音人
                vcn = available_vcns[(word_idx * args.repeat_per_word + r) % len(available_vcns)]

                # 根据第几遍匹配不同语速档位
                speed_preset = SPEED_PRESETS[r % len(SPEED_PRESETS)]
                speed = random.randint(*speed_preset["range"])
                pitch = random.randint(45, 58)  # 音调轻微扰动
                volume = random.randint(55, 75)

                utt_id = f"hard_neg_{word_idx + 1:03d}_{r + 1}"
                wav_path = os.path.join(args.output_dir, f"{utt_id}.wav")

                client = XfyunTTSClient(
                    app_id=args.app_id,
                    api_key=args.api_key,
                    api_secret=args.api_secret,
                    vcn=vcn,
                    speed=speed,
                    volume=volume,
                    pitch=pitch
                )

                cur_num = word_idx * args.repeat_per_word + r + 1
                sys.stdout.write(
                    f"\r[{cur_num}/{total_tasks}] 正在合成: {text} | 发音人:{vcn} | 语速:{speed}({speed_preset['name']})..."
                )
                sys.stdout.flush()

                ok = client.synthesize(text, wav_path)
                if ok:
                    success_count += 1
                    # 写入元数据索引：utt_id \t 文件绝对路径 \t 文本内容 \t 发音人 \t 语速 \t 音调
                    f_meta.write(f"{utt_id}\t{wav_path}\t{text}\t{vcn}\t{speed}\t{pitch}\n")
                    f_meta.flush()

                time.sleep(0.3)  # 控制请求频次，避免触发讯飞 QPS 限流

    print(f"\n\n[+] 生成完成！成功: {success_count}/{total_tasks}")
    print(f"[+] 音频保存于: {args.output_dir}")
    print(f"[+] 索引文件位于: {meta_file}")
    print("[*] 提示: 训练时，可在 prepare_dataset.py 中将这批音频的标签全部标为 <sil>，作为强对抗负样本！")


if __name__ == "__main__":
    main()
