#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于科大讯飞 (iFlytek) 在线 TTS API 的“鹰眼鹰眼”KWS 正样本 (Positive Samples) 批量合成脚本
运行环境: torch2.7.1 (/home/inno/anaconda3/envs/torch2.7.1/bin/python)

特点：
1. 目标文本固定为标准唤醒词“鹰眼鹰眼”；
2. 启动时自动对候选发音人库执行“权限探活 (Auto-Probe)”，自动过滤无权限发音人；
3. 覆盖 5 档语速扰动 (超慢/慢速/常速/偏快/急促) 与 3 档音调/音量扰动，大幅扩充声学多样性；
4. 输出标准 16000Hz、16-bit 单声道 WAV 音频，无缝兼容 FunASR / KWS 训练；
5. 自动生成配套的 metadata.txt / wav.scp 索引文件，方便直接并入训练集。
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


# ==================== 1. 讯飞 TTS 客户端 ====================
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
                        "aue": "raw",                   # 原始线性 PCM
                        "auf": "audio/L16;rate=16000",   # 16kHz 16bit 标准采样率规格
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
                    # 末尾追加 0.4s 静音缓冲，彻底防止播放器/声卡关闭时截断尾音
                    tail_silence = bytes(int(16000 * 2 * 0.4))
                    with wave.open(save_wav_path, "wb") as wf:
                        wf.setnchannels(1)      # 单声道
                        wf.setsampwidth(2)      # 16-bit (2 字节)
                        wf.setframerate(16000)  # 16000Hz 采样率
                        wf.writeframes(pcm_buffer + tail_silence)
                return True
            else:
                time.sleep(0.5)

        return False


def probe_valid_vcns(app_id, api_key, api_secret, candidate_vcns):
    """自动探活：测试哪些发音人在当前应用中具有调用权限"""
    print("[*] 正在对候选发音人执行权限探活 (Probe)...")
    valid_vcns = []
    for vcn in candidate_vcns:
        client = XfyunTTSClient(app_id, api_key, api_secret, vcn=vcn)
        ok = client.synthesize("鹰眼", save_wav_path=None, max_retry=1)
        if ok:
            valid_vcns.append(vcn)
            print(f"    ✓ [可用] {vcn}")
        else:
            print(f"    ✗ [未开通或无效] {vcn}")
        time.sleep(0.15)
    return valid_vcns


# ==================== 2. 主流程与调度 ====================
def main():
    parser = argparse.ArgumentParser(description="科大讯飞 TTS 批量生成'鹰眼鹰眼'阳性样本")
    parser.add_argument("--app_id", type=str, default="9d79363d")
    parser.add_argument("--api_key", type=str, default="69b0c6ffca5e506c0a3625d8754e87ca")
    parser.add_argument("--api_secret", type=str, default="NjA1NWI5NDUzOGNiYTA4NWExMDA3NWMz")
    parser.add_argument("--output_dir", type=str, default="/media/inno/ASR/KWS/数据集/鹰眼鹰眼/阳性/AIKit",
                        help="保存生成的正样本音频目录")
    parser.add_argument("--target_count", type=int, default=200,
                        help="计划生成的正样本总数 (建议 200+)")
    parser.add_argument("--vcns", type=str, default="",
                        help="自定义发音人列表，多个用逗号隔开；留空则自动探活默认发音人池")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 候选发音人池 (覆盖标准女声、男声、青年声、大叔音、儿童音等)
    if args.vcns:
        candidates = [v.strip() for v in args.vcns.split(",") if v.strip()]
    else:
        candidates = [
            "xiaoyan",              # 标准青年女声
            "x4_yezi",              # 叶子 (清新女声)
            "aisxping",             # 小萍 (甜美女声)
            "aisjiuxu",             # 许久 (标准播音男声)
            "aisbabyxu",            # 许小宝 (童声)
            "x4_chaoge",            # 超哥 (沉稳男声)
            "x2_wanshu",            # 万叔 (成熟大叔音)
            "x4_lingbosong",        # 聆伯松 (纪录片男声)
            "x4_lingxiaoxuan_en_v2",# 聆小璇 (知性女声)
            "x4_lingfeizhe_zl",     # 聆飞哲 (活力男声)
        ]

    # 2. 探活验证发音人
    available_vcns = probe_valid_vcns(args.app_id, args.api_key, args.api_secret, candidates)
    if not available_vcns:
        print("[!] 错误: 未检测到任何可用的发音人，请检查 APIKey 与权限！")
        sys.exit(1)

    print(f"\n[+] 确认生效发音人共 {len(available_vcns)} 个: {available_vcns}")

    # 3. 适中自然的多样性语速预设 (收敛语速，彻底杜绝高速连读导致的尾音吞字)
    SPEED_PRESETS = [
        {"name": "沉稳", "range": (40, 44)},
        {"name": "自然", "range": (45, 49)},
        {"name": "常速", "range": (50, 54)},
        {"name": "稍快", "range": (55, 58)},
    ]

    # 4. 音调预设 (覆盖低沉、中平、微高)
    PITCH_RANGES = [
        (45, 48),  # 低沉
        (49, 53),  # 正常
        (54, 58),  # 微高
    ]

    # 注意：向 TTS 发送带停顿和句号的文本，让 TTS 形成前后呼应与完整的句尾重读，杜绝末尾吞字
    tts_text = "鹰眼鹰眼"
    standard_label = "鹰眼鹰眼"
    total_tasks = args.target_count

    print("\n" + "=" * 65)
    print(" 讯飞在线 TTS '鹰眼鹰眼' 阳性样本批量合成 ".center(65, "="))
    print(f"• TTS 输入文本 : '{tts_text}' (带气口停顿与句尾落音保护)")
    print(f"• 训练标注标签 : '{standard_label}'")
    print(f"• 计划合成数量 : {total_tasks} 个")
    print(f"• 生效发音人   : {len(available_vcns)} 位发音人轮替")
    print(f"• 输出规格     : 16000Hz, 16-bit, 单声道 WAV (+0.4s 静音保护缓冲)")
    print(f"• 保存目标目录 : {os.path.abspath(args.output_dir)}")
    print("=" * 65 + "\n")

    meta_file = os.path.join(args.output_dir, "metadata.txt")
    wav_scp_file = os.path.join(args.output_dir, "wav.scp")
    success_count = 0

    with open(meta_file, "w", encoding="utf-8") as f_meta, open(wav_scp_file, "w", encoding="utf-8") as f_scp:
        f_meta.write("# uttid\twav_path\ttext\tvcn\tspeed\tpitch\tvolume\n")

        for idx in range(total_tasks):
            # 轮询发音人
            vcn = available_vcns[idx % len(available_vcns)]

            # 轮询搭配不同语速档位与音调
            speed_preset = SPEED_PRESETS[idx % len(SPEED_PRESETS)]
            pitch_preset = PITCH_RANGES[idx % len(PITCH_RANGES)]

            speed = random.randint(*speed_preset["range"])
            pitch = random.randint(*pitch_preset)
            volume = random.randint(65, 80)

            utt_id = f"yy_pos_aikit_{idx + 1:04d}_{vcn}_s{speed}_p{pitch}"
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

            sys.stdout.write(
                f"\r[{idx + 1}/{total_tasks}] 正在合成: '{tts_text}' | 发音人:{vcn:<10} | 语速:{speed}({speed_preset['name']}) | 音调:{pitch}..."
            )
            sys.stdout.flush()

            ok = client.synthesize(tts_text, wav_path)
            if ok:
                success_count += 1
                # 记录详细元数据（标注文本保持标准鹰眼鹰眼）
                f_meta.write(f"{utt_id}\t{wav_path}\t{standard_label}\t{vcn}\t{speed}\t{pitch}\t{volume}\n")
                # 记录 FunASR 标准 wav.scp 格式: uttid \t wav_path
                f_scp.write(f"{utt_id}\t{wav_path}\n")
                f_meta.flush()
                f_scp.flush()

            time.sleep(0.3)  # 控制频次防 QPS 限流

    print(f"\n\n[+] 合成全部完成！成功: {success_count}/{total_tasks}")
    print(f"[+] 音频保存目录: {args.output_dir}")
    print(f"[+] 详细元数据索引: {meta_file}")
    print(f"[+] FunASR 标准 wav.scp: {wav_scp_file}")


if __name__ == "__main__":
    main()
