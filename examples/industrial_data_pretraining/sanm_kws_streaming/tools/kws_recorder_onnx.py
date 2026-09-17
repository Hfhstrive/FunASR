#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 ONNX 的 KWS 唤醒词实时监听与录音脚本（无 funasr、无 PyTorch 依赖）。

依赖:
    pip install onnxruntime kaldi-native-fbank sounddevice soundfile numpy
    # 可选按键触发: pip install pynput

模型信息（来自训练配置 sanm_6e_320_256_fdim40_t2602.yaml）:
    - FBank: 40 维 Mel, 25ms 窗, 10ms 移, 16kHz
    - LFR:   lfr_m=7, lfr_n=3
    - 特征维度: 40 × 7 = 280
    - CMVN:  am.mvn.dim40_l3r3
    - 词表:  tokens_2602.txt, blank id=0, <unk> id=2601
    - 唤醒词 "鹰眼鹰眼" 编码为 [2601, 2601, 2601, 2601]
"""

import os
import sys
import time
import queue
import argparse
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    sd = None

import onnxruntime as ort
import kaldi_native_fbank as knf


# ==================== FBank + LFR + CMVN 前端 ====================

def load_cmvn(cmvn_path):
    """解析 Kaldi 格式的 am.mvn，返回 (neg_mean, inv_std)"""
    neg_mean = None
    inv_std = None
    with open(cmvn_path) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]
            t = list(map(float, t))
            if neg_mean is None:
                neg_mean = np.array(t, dtype=np.float32)
            else:
                inv_std = np.array(t, dtype=np.float32)
    return neg_mean, inv_std


class KwsFrontend:
    """FBank(40) + LFR(7,6) + CMVN，输出 (1, T, 280)"""

    def __init__(self, cmvn_path, sample_rate=16000,
                 n_mels=40, lfr_m=7, lfr_n=6):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.neg_mean, self.inv_std = load_cmvn(cmvn_path)
        print(f"[+] CMVN 已加载，维度: {self.neg_mean.shape[0]}")
        expected = n_mels * lfr_m
        if self.neg_mean.shape[0] != expected:
            print(f"[!] 警告: CMVN 维度 {self.neg_mean.shape[0]} "
                  f"与预期 {expected} 不一致")

    def extract(self, waveform: np.ndarray):
        """输入 float32 [-1,1] 波形，返回 (1, T, D*lfr_m) 特征；帧数不足返回 None"""
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.sample_rate
        opts.mel_opts.num_bins = self.n_mels

        online_fbank = knf.OnlineFbank(opts)
        # Kaldi 前端要求波形乘以 32768
        online_fbank.accept_waveform(
            self.sample_rate, (waveform * 32768).tolist()
        )
        online_fbank.input_finished()

        features = np.stack([
            online_fbank.get_frame(i)
            for i in range(online_fbank.num_frames_ready)
        ])

        # LFR: 堆叠相邻 lfr_m 帧，步长 lfr_n
        T = (features.shape[0] - self.lfr_m) // self.lfr_n + 1
        if T <= 0:
            return None
        features = np.lib.stride_tricks.as_strided(
            features,
            shape=(T, features.shape[1] * self.lfr_m),
            strides=((self.lfr_n * features.shape[1]) * 4, 4),
        )

        # CMVN: x = (x + neg_mean) * inv_std
        features = (features + self.neg_mean) * self.inv_std
        return features[None, ...].astype(np.float32)


# ==================== ONNX 唤醒词检测器 ====================

class OnnxKWSDetector:
    def __init__(self, onnx_path, cmvn_path, keywords="鹰眼鹰眼",
                 score_thresh=0.6, n_mels=40, lfr_m=7, lfr_n=6,
                 blank_id=0, unk_id=2601, keyword_len=4):
        self.keywords = keywords
        self.score_thresh = score_thresh
        self.blank_id = blank_id
        self.unk_id = unk_id
        self.keyword_len = keyword_len

        print(f"[*] 加载 ONNX 模型: {onnx_path}")
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        print(f"[*] 输入: {self.input_names}")
        for i in self.session.get_inputs():
            print(f"    {i.name}: shape={i.shape}")
        for o in self.session.get_outputs():
            print(f"    输出 {o.name}: shape={o.shape}")

        # 关键词 Token 映射:
        # 在底模词表中，“鹰”不在词表，模型训练收敛将其映射为同音字“英”(id=2236)；“眼”的 id=2037
        if keywords == "鹰眼鹰眼":
            self.keyword_ids = [2236, 2037, 2236, 2037]
            print(f"[+] 关键词 '{keywords}' 映射为同音词表序列: 英(2236) 眼(2037) 英(2236) 眼(2037)")
        else:
            self.keyword_ids = [self.unk_id] * self.keyword_len
            print(f"[+] 关键词 '{keywords}' 用 <unk>(id={unk_id}) × {keyword_len} 表示")
        print(f"[+] 目标 token 序列: {self.keyword_ids}")

        self.frontend = KwsFrontend(
            cmvn_path, n_mels=n_mels, lfr_m=lfr_m, lfr_n=lfr_n
        )

    def detect(self, audio_data: np.ndarray, debug=False):
        """返回 (is_detected, keyword, score)"""
        if len(audio_data) < 1600:
            return False, None, 0.0

        speech = self.frontend.extract(audio_data)
        if speech is None:
            return False, None, 0.0

        speech_lengths = np.array([speech.shape[1]], dtype=np.int32)

        try:
            outputs = self.session.run(
                None,
                {self.input_names[0]: speech,
                 self.input_names[1]: speech_lengths}
            )
        except Exception as e:
            print(f"[!] ONNX 推理失败: {e}")
            return False, None, 0.0

        # ONNX 导出的输出已经过 Softmax 概率化
        probs = outputs[0][0]  # (T, 2602)

        # ---- CTC greedy 解码 ----
        preds = np.argmax(probs, axis=-1)  # (T,)
        collapsed = []
        prev = -1
        for p in preds:
            if p != prev:
                if p != self.blank_id:
                    collapsed.append(int(p))
                prev = int(p)

        if debug:
            print(f"[DEBUG] 折叠序列: {collapsed}")
            print(f"[DEBUG] 目标: {self.keyword_ids}")

        # ---- 关键词子序列匹配 ----
        kws_ids = self.keyword_ids
        n = len(kws_ids)
        found = False
        best_score = 0.0

        for start in range(len(collapsed) - n + 1):
            if collapsed[start:start + n] == kws_ids:
                found = True
                # 统计匹配 token 的最大置信度
                match_scores = [float(np.max(probs[:, tid])) for tid in set(kws_ids)]
                score = float(np.mean(match_scores)) if match_scores else 0.0
                best_score = max(best_score, score)

        if found and best_score >= self.score_thresh:
            return True, self.keywords, best_score
        return False, None, 0.0


# ==================== 录音管理器 ====================

class AudioStreamRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.1,
                 silence_duration=3.0, silence_thresh=0.015,
                 save_dir="./records", mic_index=None):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.silence_duration = silence_duration
        self.silence_thresh = silence_thresh
        self.save_dir = save_dir
        self.mic_index = mic_index
        os.makedirs(self.save_dir, exist_ok=True)
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[!] 音频状态异常: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy().flatten())

    def calibrate_noise(self, duration=1.0):
        print(f"[*] 采集环境噪音 ({duration}s)，请保持安静...")
        rec = sd.rec(int(duration * self.sample_rate),
                     samplerate=self.sample_rate, channels=1,
                     dtype="float32", device=self.mic_index)
        sd.wait()
        rms = np.sqrt(np.mean(rec.flatten() ** 2))
        adapted = max(0.008, min(0.05, rms * 3.0))
        print(f"[+] 环境 RMS: {rms:.5f}, 静音阈值: {adapted:.5f}")
        self.silence_thresh = adapted


# ==================== 键盘触发 ====================

class KeyboardTrigger:
    def __init__(self, hotkey="s", enabled=True):
        self.hotkey = hotkey.lower()
        self.enabled = enabled
        self.triggered = False
        self._listener = None

    def start(self):
        if not self.enabled:
            return
        try:
            from pynput import keyboard
        except ImportError:
            print("[!] 未安装 pynput，按键触发不可用。pip install pynput",
                  file=sys.stderr)
            self.enabled = False
            return

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char and \
                        key.char.lower() == self.hotkey:
                    self.triggered = True
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()
        print(f"[+] 按键触发已启用：按 '{self.hotkey.upper()}' 开始录音")

    def consume(self):
        if self.triggered:
            self.triggered = False
            return True
        return False

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass


# ==================== 仿真模式 ====================

def run_file_simulation(wav_path, detector, recorder, silence_duration,
                        debug=False):
    """无麦克风环境：按 0.1s 切片喂 wav，复用同一套状态机"""
    print(f"\n[*] 仿真模式：{wav_path}")

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        print(f"[*] 采样率 {sr}Hz ≠ 16000Hz，执行高质量多相重采样...")
        try:
            import scipy.signal
            data = scipy.signal.resample_poly(data, 16000, sr).astype(np.float32)
        except ImportError:
            tgt_len = int(len(data) * 16000 / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, tgt_len),
                np.arange(len(data)), data
            ).astype(np.float32)

    chunk_size = recorder.chunk_samples
    num_chunks = len(data) // chunk_size + 1

    state = "LISTENING"
    kws_buffer = np.zeros(0, dtype=np.float32)
    recorded_frames = []
    last_voice_time = 0.0
    accumulated = 0
    sim_time = 0.0
    wake_count = 0
    step_samples = int(16000 * 0.1)
    window_samples = int(16000 * 1.8)

    print(f"[*] 音频总时长: {len(data)/16000:.2f}s，开始仿真...")

    for i in range(num_chunks):
        chunk = data[i * chunk_size: (i + 1) * chunk_size]
        if len(chunk) == 0:
            break
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        sim_time += len(chunk) / 16000
        chunk_rms = np.sqrt(np.mean(chunk ** 2))

        if state == "LISTENING":
            kws_buffer = np.concatenate([kws_buffer, chunk])
            accumulated += len(chunk)
            if len(kws_buffer) > window_samples:
                kws_buffer = kws_buffer[-window_samples:]

            if accumulated >= step_samples and len(kws_buffer) >= 12800:
                accumulated = 0
                detected, kw, score = detector.detect(kws_buffer, debug=debug)
                if detected:
                    wake_count += 1
                    print(f"\n[{sim_time:.2f}s] 🎉 第 {wake_count} 次唤醒 "
                          f"'{kw}' (置信度: {score:.4f})")
                    state = "RECORDING"
                    recorded_frames = []
                    last_voice_time = sim_time
                    kws_buffer = np.zeros(0, dtype=np.float32)

        elif state == "RECORDING":
            recorded_frames.append(chunk)
            if chunk_rms > recorder.silence_thresh:
                last_voice_time = sim_time
            else:
                elapsed = sim_time - last_voice_time
                if elapsed >= silence_duration:
                    print(f"\n[{sim_time:.2f}s] ⏹️ 静音 {silence_duration}s，结束录音")
                    total = np.concatenate(recorded_frames)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{ts}.wav"
                    fpath = os.path.join(recorder.save_dir, fname)
                    sf.write(fpath, total, 16000)
                    print(f"💾 已保存: {fpath} ({len(total)/16000:.2f}s)")
                    state = "LISTENING"
                    kws_buffer = np.zeros(0, dtype=np.float32)
                    recorded_frames = []

    if state == "RECORDING" and recorded_frames:
        print(f"\n[{sim_time:.2f}s] ℹ️ 音频流结束，结算录音...")
        total = np.concatenate(recorded_frames)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(recorder.save_dir, f"{ts}.wav")
        sf.write(fpath, total, 16000)
        print(f"💾 已保存: {fpath} ({len(total)/16000:.2f}s)")

    print(f"\n[*] 仿真结束，共检测到 {wake_count} 次唤醒。")


# ==================== 实时麦克风主流程 ====================

def run_realtime(args, detector, recorder, keyboard_trigger, debug=False):
    state = "LISTENING"
    kws_buffer = np.zeros(0, dtype=np.float32)
    recorded_frames = []
    last_voice_time = 0.0
    accumulated = 0
    step_samples = int(16000 * 0.1)
    window_samples = int(16000 * 1.8)

    print(f"👂 [监听中] 请说出 '{args.keywords}' 或按 S 键...")

    with sd.InputStream(
        samplerate=16000, channels=1, dtype="float32",
        device=recorder.mic_index, callback=recorder._audio_callback,
        blocksize=recorder.chunk_samples,
    ):
        try:
            while True:
                try:
                    chunk = recorder.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if state == "LISTENING" and keyboard_trigger.consume():
                        print("\n⌨️  [手动触发] 开始录音...")
                        print("🔴 [录音中] 静音 3 秒自动停止...")
                        state = "RECORDING"
                        recorded_frames = []
                        last_voice_time = time.time()
                        kws_buffer = np.zeros(0, dtype=np.float32)
                    continue

                chunk_rms = np.sqrt(np.mean(chunk ** 2))

                if state == "LISTENING":
                    if keyboard_trigger.consume():
                        print("\n⌨️  [手动触发] 开始录音...")
                        print("🔴 [录音中] 静音 3 秒自动停止...")
                        state = "RECORDING"
                        recorded_frames = []
                        last_voice_time = time.time()
                        kws_buffer = np.zeros(0, dtype=np.float32)
                        continue

                    kws_buffer = np.concatenate([kws_buffer, chunk])
                    accumulated += len(chunk)
                    if len(kws_buffer) > window_samples:
                        kws_buffer = kws_buffer[-window_samples:]

                    if accumulated >= step_samples and len(kws_buffer) >= 12800:
                        accumulated = 0
                        detected, kw, score = detector.detect(
                            kws_buffer, debug=debug
                        )
                        if detected:
                            print(f"\n🎉 [唤醒] '{kw}' (置信度: {score:.4f})")
                            print("🔴 [录音中] 静音 3 秒自动停止...")
                            state = "RECORDING"
                            recorded_frames = []
                            last_voice_time = time.time()
                            kws_buffer = np.zeros(0, dtype=np.float32)

                elif state == "RECORDING":
                    recorded_frames.append(chunk)
                    now = time.time()

                    if chunk_rms > recorder.silence_thresh:
                        last_voice_time = now
                        sys.stdout.write(
                            f"\r🎙️  收音中... [音量: {chunk_rms:.4f}]"
                        )
                        sys.stdout.flush()
                    else:
                        elapsed = now - last_voice_time
                        sys.stdout.write(
                            f"\r⏳ 静音 {elapsed:.1f}s / "
                            f"{recorder.silence_duration}s"
                        )
                        sys.stdout.flush()

                        if elapsed >= recorder.silence_duration:
                            print("\n\n⏹️  [录音结束]")
                            total = np.concatenate(recorded_frames)

                            trim = int(16000 * max(
                                0, recorder.silence_duration - 0.5
                            ))
                            if len(total) > trim + 8000:
                                save_audio = total[:-trim]
                            else:
                                save_audio = total

                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fname = f"{ts}.wav"
                            fpath = os.path.join(recorder.save_dir, fname)
                            sf.write(fpath, save_audio, 16000)

                            dur = len(save_audio) / 16000
                            print(f"💾 已保存: {fpath} ({dur:.2f}s)")

                            state = "LISTENING"
                            kws_buffer = np.zeros(0, dtype=np.float32)
                            recorded_frames = []
                            accumulated = 0
                            print(f"\n👂 [监听中] 请说出 "
                                  f"'{args.keywords}' 或按 S 键...")

        except KeyboardInterrupt:
            keyboard_trigger.stop()
            print("\n\n[👋] 已退出。")


# ==================== 主入口 ====================

def run(args):
    if sd is None and not args.test_wav:
        print("[!] sounddevice 不可用。pip install sounddevice",
              file=sys.stderr)
        sys.exit(1)

    detector = OnnxKWSDetector(
        onnx_path=args.onnx_model,
        cmvn_path=args.cmvn_path,
        keywords=args.keywords,
        score_thresh=args.score_thresh,
        n_mels=args.n_mels,
        lfr_m=args.lfr_m,
        lfr_n=args.lfr_n,
        blank_id=args.blank_id,
        unk_id=args.unk_id,
        keyword_len=args.keyword_len,
    )

    recorder = AudioStreamRecorder(
        sample_rate=16000,
        silence_duration=args.silence_duration,
        silence_thresh=args.silence_thresh,
        save_dir=args.save_dir,
        mic_index=args.mic_index,
    )

    # 仿真模式
    if args.test_wav:
        run_file_simulation(
            args.test_wav, detector, recorder,
            args.silence_duration, debug=args.debug
        )
        return

    # 实时麦克风模式
    keyboard_trigger = KeyboardTrigger(
        hotkey=args.hotkey, enabled=args.enable_hotkey
    )

    if args.auto_calibrate:
        recorder.calibrate_noise()

    keyboard_trigger.start()

    print("\n" + "=" * 60)
    print("  ONNX KWS 监听系统已就绪（无 funasr 依赖）")
    print(f"  • 唤醒词       : {args.keywords} (编码: <unk>×{args.keyword_len})")
    print(f"  • 手动按键     : {'S' if args.enable_hotkey else '未启用'}")
    print(f"  • 静音停止     : {args.silence_duration}s")
    print(f"  • 保存目录     : {os.path.abspath(args.save_dir)}")
    print("=" * 60 + "\n")

    run_realtime(args, detector, recorder, keyboard_trigger,
                 debug=args.debug)


def parse_args():
    p = argparse.ArgumentParser(
        description="ONNX KWS 唤醒词监听与录音（无 funasr 依赖）"
    )
    p.add_argument("--onnx_model", type=str,
                   default="/media/inno/work_dirs/ASR/kws_yingyan/onnx/encoder.onnx",
                   help="encoder.onnx 路径")
    p.add_argument("--cmvn_path", type=str,
                   default="/media/inno/work_dirs/ASR/kws_yingyan/am.mvn.dim40_l3r3",
                   help="am.mvn.dim40_l3r3 路径")
    p.add_argument("--keywords", type=str, default="鹰眼鹰眼")
    p.add_argument("--score_thresh", type=float, default=0.6)
    p.add_argument("--n_mels", type=int, default=40)
    p.add_argument("--lfr_m", type=int, default=7)
    p.add_argument("--lfr_n", type=int, default=6)
    p.add_argument("--blank_id", type=int, default=0,
                   help="CTC blank 的 id")
    p.add_argument("--unk_id", type=int, default=2601,
                   help="<unk> 的 id")
    p.add_argument("--keyword_len", type=int, default=4,
                   help="关键词对应的 token 数（鹰眼鹰眼=4）")
    p.add_argument("--silence_duration", type=float, default=3.0)
    p.add_argument("--silence_thresh", type=float, default=0.015)
    p.add_argument("--auto_calibrate", action="store_true")
    p.add_argument("--save_dir", type=str, default="/media/inno/output/ASR/唤醒词/唤醒词验证/onnx/")
    p.add_argument("--mic_index", type=int, default=None)
    p.add_argument("--enable_hotkey", action="store_true")
    p.add_argument("--hotkey", type=str, default="s")
    # p.add_argument("--test_wav", type=str, default=None,
    # p.add_argument("--test_wav", type=str, default="/media/inno/ASR/kws_root/test/唤醒词-鹰眼鹰眼.wav",
    p.add_argument("--test_wav", type=str, default="/media/inno/ASR/唤醒词/数据集/阴性/肠镜/66_70.wav",
                   help="仿真模式：传入 wav 路径；不传则走实时麦克风")
    p.add_argument("--debug", action="store_true",
                   help="打印 CTC 解码调试信息")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())