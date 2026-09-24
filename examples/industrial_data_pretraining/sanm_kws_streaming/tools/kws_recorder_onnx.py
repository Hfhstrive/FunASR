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
                 score_thresh=0.8, min_span=0.40, max_span=1.35, max_gap=0.40, min_token_score=0.65,
                 n_mels=40, lfr_m=7, lfr_n=6, blank_id=0, unk_id=2601, keyword_len=4):
        self.keywords = keywords
        self.score_thresh = score_thresh
        self.min_span = min_span
        self.max_span = max_span
        self.max_gap = max_gap
        self.min_token_score = min_token_score
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
        print(f"[+] 时空物理约束已启用: 总时长[{self.min_span:.2f}s~{self.max_span:.2f}s], 最大字间距<={self.max_gap:.2f}s, 单字最低分>={self.min_token_score:.2f}")

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

        # ---- CTC greedy 解码并记录带时间戳与概率的活跃 Token ----
        preds = np.argmax(probs, axis=-1)  # (T,)
        active_tokens = []
        for frame_idx, p in enumerate(preds):
            if p != self.blank_id:
                active_tokens.append((frame_idx, int(p), float(probs[frame_idx, p])))

        # 连续相同 token 折叠去重，保留时间戳
        collapsed_with_time = []
        for item in active_tokens:
            if not collapsed_with_time or collapsed_with_time[-1][1] != item[1]:
                collapsed_with_time.append(item)

        collapsed_ids = [it[1] for it in collapsed_with_time]

        if debug:
            print(f"[DEBUG] 折叠序列: {collapsed_ids}")
            print(f"[DEBUG] 目标: {self.keyword_ids}")

        # ---- 关键词子序列匹配 + 时空物理强约束校验 ----
        kws_ids = self.keyword_ids
        n = len(kws_ids)
        frame_time_step = 0.01 * self.frontend.lfr_n  # 默认 0.01 * 6 = 0.06s (60ms一帧)

        for start in range(len(collapsed_ids) - n + 1):
            if collapsed_ids[start : start + n] == kws_ids:
                sub = collapsed_with_time[start : start + n]
                frames = [it[0] for it in sub]
                scores = [it[2] for it in sub]

                # 约束 1: 音节总时长物理跨度 (4个字人声正常耗时约 0.40s ~ 1.35s)
                span_sec = (frames[-1] - frames[0]) * frame_time_step

                # 约束 2: 相邻字最大停顿时间 (防止长句中东拼西凑)
                gaps = [(frames[j + 1] - frames[j]) * frame_time_step for j in range(n - 1)]
                max_gap_sec = max(gaps) if gaps else 0.0

                # 约束 3: 独立字置信度与平均置信度校验 (防止单个高分拉高均分)
                min_token_sc = min(scores)
                avg_sc = float(np.mean(scores))

                if debug:
                    print(f"[DEBUG 物理约束] 候选 '{self.keywords}': 跨度={span_sec:.2f}s, 最大字间距={max_gap_sec:.2f}s, 单字最低={min_token_sc:.3f}, 均分={avg_sc:.3f}")

                # 物理强约束逐条过滤校验
                if not (self.min_span <= span_sec <= self.max_span):
                    if debug:
                        print(f"[DEBUG 物理约束拒绝] 发音总时长跨度 {span_sec:.2f}s 不在 [{self.min_span:.2f}s, {self.max_span:.2f}s] 区间")
                    continue
                if max_gap_sec > self.max_gap:
                    if debug:
                        print(f"[DEBUG 物理约束拒绝] 最大字间停顿 {max_gap_sec:.2f}s 超过允许上限 {self.max_gap:.2f}s")
                    continue
                if min_token_sc < self.min_token_score:
                    if debug:
                        print(f"[DEBUG 物理约束拒绝] 单字最低置信度 {min_token_sc:.3f} 低于门限 {self.min_token_score:.2f}")
                    continue
                if avg_sc < self.score_thresh:
                    if debug:
                        print(f"[DEBUG 物理约束拒绝] 平均置信度 {avg_sc:.3f} 低于门限 {self.score_thresh:.2f}")
                    continue

                # 全部物理约束与置信度门限校验通过
                return True, self.keywords, avg_sc

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


def trim_trailing_silence(audio_data: np.ndarray, sample_rate: int = 16000,
                          silence_thresh: float = 0.015,
                          safe_margin: float = 0.8,
                          chunk_ms: int = 40) -> np.ndarray:
    """
    智能反向回溯定位真实发音终点，并保留充裕的尾音安全余量 (默认 0.8 秒)。

    原理：
    从音频尾部倒序向前按 40ms 小窗扫描，检测能量高于弱音门限 (silence_thresh * 0.5) 的真实语音截止位置。
    从该位置向后追加 safe_margin 秒的安全缓冲静音，多余无用底噪予以切除。
    彻底避免硬切造成的吞字和尾音截断，同时留出足够的 ASR 空白闭合帧。
    """
    if len(audio_data) == 0:
        return audio_data

    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    voice_floor = max(0.008, silence_thresh * 0.8)
    num_chunks = len(audio_data) // chunk_size

    last_voice_idx = -1
    for i in range(num_chunks - 1, -1, -1):
        seg = audio_data[i * chunk_size : (i + 1) * chunk_size]
        if np.sqrt(np.mean(seg ** 2)) >= voice_floor:
            last_voice_idx = (i + 1) * chunk_size
            break

    if last_voice_idx == -1:
        min_keep = int(sample_rate * 1.0)
        return audio_data[-min_keep:] if len(audio_data) > min_keep else audio_data

    cutoff = min(len(audio_data), last_voice_idx + int(sample_rate * safe_margin))
    cutoff = max(int(sample_rate * 0.5), cutoff)
    cutoff = min(len(audio_data), cutoff)
    return audio_data[:cutoff]


# ==================== 仿真模式 ====================

def run_file_simulation(wav_path, detector, recorder, silence_duration,
                        debounce_hits=1, safe_margin=0.8, debug=False):
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
    consecutive_hits = 0
    step_samples = int(16000 * 0.1)
    window_samples = int(16000 * 1.8)

    print(f"[*] 音频总时长: {len(data)/16000:.2f}s，开始仿真 (防抖确认: {debounce_hits} 次)...")

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
                    consecutive_hits += 1
                    if consecutive_hits >= debounce_hits:
                        wake_count += 1
                        print(f"\n[{sim_time:.2f}s] 🎉 第 {wake_count} 次唤醒 "
                              f"'{kw}' (置信度: {score:.4f}, 防抖: {consecutive_hits}/{debounce_hits})")
                        state = "RECORDING"
                        recorded_frames = []
                        last_voice_time = sim_time
                        kws_buffer = np.zeros(0, dtype=np.float32)
                        consecutive_hits = 0
                else:
                    consecutive_hits = 0

        elif state == "RECORDING":
            recorded_frames.append(chunk)
            if chunk_rms > recorder.silence_thresh:
                last_voice_time = sim_time
            else:
                elapsed = sim_time - last_voice_time
                if elapsed >= silence_duration:
                    print(f"\n[{sim_time:.2f}s] ⏹️ 静音 {silence_duration}s，结束录音")
                    total = np.concatenate(recorded_frames)
                    save_audio = trim_trailing_silence(
                        total,
                        sample_rate=16000,
                        silence_thresh=recorder.silence_thresh,
                        safe_margin=safe_margin
                    )
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{ts}.wav"
                    fpath = os.path.join(recorder.save_dir, fname)
                    sf.write(fpath, save_audio, 16000)
                    print(f"💾 已保存: {fpath} ({len(save_audio)/16000:.2f}s，含 {safe_margin}s 尾音安全余量)")
                    state = "LISTENING"
                    kws_buffer = np.zeros(0, dtype=np.float32)
                    recorded_frames = []
                    consecutive_hits = 0

    if state == "RECORDING" and recorded_frames:
        print(f"\n[{sim_time:.2f}s] ℹ️ 音频流结束，结算录音...")
        total = np.concatenate(recorded_frames)
        save_audio = trim_trailing_silence(
            total,
            sample_rate=16000,
            silence_thresh=recorder.silence_thresh,
            safe_margin=safe_margin
        )
        ts = time.strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(recorder.save_dir, f"{ts}.wav")
        sf.write(fpath, save_audio, 16000)
        print(f"💾 已保存: {fpath} ({len(save_audio)/16000:.2f}s)")

    print(f"\n[*] 仿真结束，共检测到 {wake_count} 次唤醒。")


# ==================== 实时麦克风主流程 ====================

def run_realtime(args, detector, recorder, keyboard_trigger, debug=False):
    state = "LISTENING"
    kws_buffer = np.zeros(0, dtype=np.float32)
    recorded_frames = []
    last_voice_time = 0.0
    accumulated = 0
    consecutive_hits = 0
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
                        consecutive_hits = 0
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
                        consecutive_hits = 0
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
                            consecutive_hits += 1
                            if consecutive_hits >= args.debounce_hits:
                                print(f"\n🎉 [唤醒] '{kw}' (置信度: {score:.4f}, 防抖: {consecutive_hits}/{args.debounce_hits})")
                                print("🔴 [录音中] 静音 3 秒自动停止...")
                                state = "RECORDING"
                                recorded_frames = []
                                last_voice_time = time.time()
                                kws_buffer = np.zeros(0, dtype=np.float32)
                                consecutive_hits = 0
                        else:
                            consecutive_hits = 0

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

                            # 智能反向回溯定位真实语音终点，并保留 safe_margin (默认 0.8s) 安全尾音
                            save_audio = trim_trailing_silence(
                                total,
                                sample_rate=16000,
                                silence_thresh=recorder.silence_thresh,
                                safe_margin=args.safe_margin
                            )

                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fname = f"{ts}.wav"
                            fpath = os.path.join(recorder.save_dir, fname)
                            sf.write(fpath, save_audio, 16000)

                            dur = len(save_audio) / 16000
                            print(f"💾 已保存: {fpath} ({dur:.2f}s，含 {args.safe_margin}s 安全尾音缓冲)")

                            state = "LISTENING"
                            kws_buffer = np.zeros(0, dtype=np.float32)
                            recorded_frames = []
                            accumulated = 0
                            consecutive_hits = 0
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
        min_span=args.min_span,
        max_span=args.max_span,
        max_gap=args.max_gap,
        min_token_score=args.min_token_score,
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
            args.silence_duration, debounce_hits=args.debounce_hits,
            safe_margin=args.safe_margin, debug=args.debug
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
    print("  ONNX KWS 监听系统已就绪（无 funasr 依赖 + 时空物理强约束）")
    print(f"  • 唤醒词       : {args.keywords}")
    print(f"  • 置信度阈值   : {args.score_thresh} (单字最低门限: {args.min_token_score})")
    print(f"  • 时空物理约束 : 4字时长[{args.min_span}s~{args.max_span}s], 最大字间停顿<={args.max_gap}s, 防抖确认={args.debounce_hits}次")
    print(f"  • 尾音安全保护 : 保留 {args.safe_margin} 秒弱音余量 (双阈值迟滞 + 智能反向回溯，绝不截断尾音)")
    print(f"  • 手动按键     : {'S' if args.enable_hotkey else '未启用'}")
    print(f"  • 静音停止     : {args.silence_duration}s")
    print(f"  • 保存目录     : {os.path.abspath(args.save_dir)}")
    print("=" * 60 + "\n")

    run_realtime(args, detector, recorder, keyboard_trigger,
                 debug=args.debug)


def parse_args():
    p = argparse.ArgumentParser(
        description="ONNX KWS 唤醒词监听与录音（无 funasr 依赖 + 时空物理强约束）"
    )
    p.add_argument("--onnx_model", type=str,
                   default="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3/onnx/encoder_quant.onnx",
                   help="encoder.onnx 路径")
    p.add_argument("--cmvn_path", type=str,
                   default="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3/am.mvn.dim40_l3r3",
                   help="am.mvn.dim40_l3r3 路径")
    p.add_argument("--keywords", type=str, default="鹰眼鹰眼")
    p.add_argument("--score_thresh", type=float, default=0.8,
                   help="平均置信度门限 (默认: 0.8)")
    p.add_argument("--min_token_score", type=float, default=0.65,
                   help="每个音节独立最低置信度门限 (默认: 0.65)")
    p.add_argument("--min_span", type=float, default=0.40,
                   help="4字最短发音总时长秒数 (默认: 0.40s)")
    p.add_argument("--max_span", type=float, default=1.35,
                   help="4字最长发音总时长秒数 (默认: 1.35s)")
    p.add_argument("--max_gap", type=float, default=0.50,
                   help="相邻音节间最大允许停顿秒数 (默认: 0.50s)")
    p.add_argument("--debounce_hits", type=int, default=1,
                   help="防抖连续命中次数确认 (默认: 1，极嘈杂环境可设为 2)")
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
    p.add_argument("--safe_margin", type=float, default=0.8,
                   help="真实语音截止后保留的尾音安全余量秒数 (默认: 0.8s)")
    p.add_argument("--silence_thresh", type=float, default=0.015)
    p.add_argument("--auto_calibrate", action="store_true")
    p.add_argument("--save_dir", type=str, default="/media/inno/output/ASR/唤醒词/v3/onnx/")
    p.add_argument("--mic_index", type=int, default=None)
    p.add_argument("--enable_hotkey", action="store_true")
    p.add_argument("--hotkey", type=str, default="s")
    p.add_argument("--test_wav", type=str, default="/media/inno/ASR/KWS/test/唤醒词-鹰眼鹰眼.wav",
                   help="仿真模式：传入 wav 路径；不传则走实时麦克风")
    p.add_argument("--debug", action="store_true",
                   help="打印 CTC 解码调试信息")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())