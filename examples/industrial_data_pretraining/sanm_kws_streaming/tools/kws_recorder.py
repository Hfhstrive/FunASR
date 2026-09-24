#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 FunASR 唤醒词模型的实时麦克风监听与录音脚本。

核心功能：
1. 实时监听唤醒词（默认："你好小云"）；
2. 听到唤醒词后自动启动录音功能；
3. 录音过程中实时检测声音，连续 3 秒没有收音（静音）时自动停止录音；
4. 录音自动保存为 WAV 音频文件；
5. 支持可选参数在录音结束后自动进行 ASR 语音识别转写；
6. 自动重新进入监听状态，循环等待下一次唤醒。
7. 【新增】支持按热键（默认 S）手动触发录音，用于现场唤醒失灵时兜底。
"""

import os
import sys
import time
import re
import queue
import argparse
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    sd = None

import math
from funasr import AutoModel
import funasr.utils.kws_utils as kws_utils


class SpatiotemporalConstraint:
    """时空物理强约束参数配置容器"""

    def __init__(self, min_span=0.40, max_span=1.35, max_gap=0.40,
                 min_token_score=0.65, frame_step=0.06, debug=False):
        self.min_span = min_span            # 4字最短发音总时长 (秒)
        self.max_span = max_span            # 4字最长发音总时长 (秒)
        self.max_gap = max_gap              # 相邻字间最大允许停顿 (秒)
        self.min_token_score = min_token_score  # 每个音节独立最低置信度门限
        self.frame_step = frame_step        # 帧移时间步长 (默认 60ms)
        self.debug = debug                  # 是否输出时空物理约束诊断日志


class KWSDetector:
    """集成时空物理强约束的唤醒词检测器"""

    def __init__(self, model_name="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
                 checkpoint_path=None,
                 keywords="鹰眼鹰眼", device="cuda:0", score_thresh=0.80,
                 min_span=0.40, max_span=1.35, max_gap=0.40, min_token_score=0.65,
                 debug=False):
        self.keywords = keywords
        self.score_thresh = score_thresh
        self.device = device
        self.constraint = SpatiotemporalConstraint(
            min_span=min_span,
            max_span=max_span,
            max_gap=max_gap,
            min_token_score=min_token_score,
            frame_step=0.06,
            debug=debug
        )

        # 注入带时空物理约束的 CTC 解码器逻辑
        self._install_spatiotemporal_decoder()

        print(f"[*] 正在加载唤醒词 (KWS) 模型: {model_name} (设备: {device})...")
        model_kwargs = {
            "model": model_name,
            "keywords": keywords,
            "output_dir": "./outputs/debug",
            "device": device,
            "chunk_size": [4, 8, 4],
            "encoder_chunk_look_back": 0,
            "decoder_chunk_look_back": 0,
            "disable_update": True,
            "disable_pbar": True,
        }
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] 加载微调 Checkpoint: {checkpoint_path}")
            model_kwargs["init_param"] = checkpoint_path

        self.kws_model = AutoModel(**model_kwargs)
        print(f"[+] 唤醒词模型加载完成，目标唤醒词: '{keywords}'，判定阈值: {score_thresh}")
        print(f"[+] 时空物理强约束已启用: 4字总时长[{min_span:.2f}s~{max_span:.2f}s], 最大字间停顿<={max_gap:.2f}s, 单字最低分>={min_token_score:.2f}")

    def _install_spatiotemporal_decoder(self):
        """将 FunASR 底层 KwsCtcPrefixDecoder 增强为具备帧级时空物理约束的解码器"""
        detector_self = self

        def spatiotemporal_decode_inside(decoder_inst, logits, logits_lengths):
            hyps = decoder_inst.beam_search(logits, logits_lengths, decoder_inst.keywords_idxset)
            hit_keyword = None
            hit_score = 0.0

            for one_hyp in hyps:
                prefix_ids, _, prefix_nodes = one_hyp
                for word in decoder_inst.keywords_token.keys():
                    lab = decoder_inst.keywords_token[word]['token_id']
                    offset = decoder_inst.is_sublist(prefix_ids, lab)
                    if offset != -1:
                        sub_nodes = prefix_nodes[offset : offset + len(lab)]
                        frames = [node['frame'] for node in sub_nodes]
                        token_probs = [node['prob'] for node in sub_nodes]

                        # 1. 发音总时长跨度约束 (正常人声 0.40s ~ 1.35s)
                        span_sec = (frames[-1] - frames[0]) * detector_self.constraint.frame_step

                        # 2. 相邻字间最大停顿约束 (防止非连续散字跨句拼凑)
                        gaps = [(frames[j + 1] - frames[j]) * detector_self.constraint.frame_step for j in range(len(frames) - 1)]
                        max_gap_sec = max(gaps) if gaps else 0.0

                        # 3. 单字独立最低置信度与综合置信度
                        min_tok_sc = min(token_probs)
                        prod_score = 1.0
                        for p in token_probs:
                            prod_score *= p
                        cur_hit_score = math.sqrt(prod_score)

                        if detector_self.constraint.debug:
                            print(f"[DEBUG 物理约束] 候选 '{word}': 跨度={span_sec:.2f}s, 最大字间距={max_gap_sec:.2f}s, 单字最低={min_tok_sc:.3f}, 综合分={cur_hit_score:.3f}")

                        # 严格物理校验：全部通过才认可为合法唤醒词
                        if not (detector_self.constraint.min_span <= span_sec <= detector_self.constraint.max_span):
                            if detector_self.constraint.debug:
                                print(f"[DEBUG 物理约束拒绝] 发音跨度 {span_sec:.2f}s 不在 [{detector_self.constraint.min_span:.2f}s, {detector_self.constraint.max_span:.2f}s] 区间")
                            continue
                        if max_gap_sec > detector_self.constraint.max_gap:
                            if detector_self.constraint.debug:
                                print(f"[DEBUG 物理约束拒绝] 最大字间停顿 {max_gap_sec:.2f}s 超过允许上限 {detector_self.constraint.max_gap:.2f}s")
                            continue
                        if min_tok_sc < detector_self.constraint.min_token_score:
                            if detector_self.constraint.debug:
                                print(f"[DEBUG 物理约束拒绝] 单字最低分 {min_tok_sc:.3f} 低于门限 {detector_self.constraint.min_token_score:.2f}")
                            continue
                        if cur_hit_score < detector_self.score_thresh:
                            if detector_self.constraint.debug:
                                print(f"[DEBUG 物理约束拒绝] 综合得分 {cur_hit_score:.3f} 低于门限 {detector_self.score_thresh:.2f}")
                            continue

                        # 通过全部物理与声学约束
                        return True, word, cur_hit_score

            return False, None, None

        kws_utils.KwsCtcPrefixDecoder._decode_inside = spatiotemporal_decode_inside

    def detect(self, audio_data: np.ndarray) -> tuple:
        """
        在给定的音频数据片段中检测唤醒词。
        返回: (is_detected, keyword, score)
        """
        if len(audio_data) < 1600:  # 音频过短时跳过
            return False, None, 0.0

        res = self.kws_model.generate(
            input=audio_data,
            chunk_size=[4, 8, 4],
            encoder_chunk_look_back=0,
            decoder_chunk_look_back=0,
            is_final=True,
            disable_pbar=True,
        )
        if not res or len(res) == 0:
            return False, None, 0.0

        res_text = res[0].get("text", "")
        # 解析格式: 'detected 鹰眼鹰眼 0.9917...'
        match = re.search(r"detected\s+(\S+)\s+([\d\.]+)", res_text)
        if match:
            keyword = match.group(1)
            score = float(match.group(2))
            if score >= self.score_thresh:
                return True, keyword, score
        return False, None, 0.0


class AudioStreamRecorder:
    """麦克风流式音频采集与录音状态机管理器"""

    def __init__(self, sample_rate=16000, chunk_duration=0.1,
                 window_duration=1.8, step_duration=0.1,
                 silence_duration=3.0, silence_thresh=0.015,
                 save_dir="./outputs/records", mic_index=None):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.window_samples = int(sample_rate * window_duration)
        self.step_samples = int(sample_rate * step_duration)
        self.silence_duration = silence_duration
        self.silence_thresh = silence_thresh
        self.save_dir = save_dir
        self.mic_index = mic_index

        os.makedirs(self.save_dir, exist_ok=True)
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 输入流回调"""
        if status:
            print(f"[!] 音频输入状态异常: {status}", file=sys.stderr)
        # 单通道 float32 数据
        self.audio_queue.put(indata.copy().flatten())

    def calibrate_noise(self, duration=1.0):
        """采集一段环境底噪，自适应调整静音阈值"""
        if sd is None:
            return self.silence_thresh

        print(f"[*] 正在采集环境噪音基准 ({duration}秒)，请保持安静...")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                           channels=1, dtype="float32", device=self.mic_index)
        sd.wait()
        rms = np.sqrt(np.mean(recording.flatten() ** 2))
        # 将静音阈值设置为环境底噪的 3 倍左右，且保底在 [0.008, 0.05] 之间
        adapted_thresh = max(0.008, min(0.05, rms * 3.0))
        print(f"[+] 环境噪音 RMS: {rms:.5f}, 动态设定静音判断阈值: {adapted_thresh:.5f}")
        self.silence_thresh = adapted_thresh
        return adapted_thresh


# ==== 新增：键盘手动触发辅助类 ====
class KeyboardTrigger:
    """非阻塞键盘触发器：按下指定键后置位标志，由主循环消费。

    依赖 pynput（跨平台，不需要 root）。
    在纯 SSH 无 X11 的 Linux 上可能拿不到键盘事件，见文档说明。
    """

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
            print("[!] 未安装 pynput，无法启用按键触发。请执行: pip install pynput",
                  file=sys.stderr)
            self.enabled = False
            return

        def on_press(key):
            try:
                # 普通字符键
                if hasattr(key, "char") and key.char and key.char.lower() == self.hotkey:
                    self.triggered = True
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()
        print(f"[+] 已启用按键触发：按 '{self.hotkey.upper()}' 键立即开始录音")

    def consume(self):
        """消费型读取：读到 True 后自动清零"""
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
            self._listener = None


def _start_recording(state_holder):
    """进入录音状态的公共动作（唤醒词路径和按键路径共用）"""
    state_holder["state"] = "RECORDING"
    state_holder["recorded_frames"] = []
    state_holder["last_voice_time"] = time.time()
    state_holder["record_start_time"] = time.time()
    state_holder["kws_buffer"] = np.zeros(0, dtype=np.float32)


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


def generate_timestamp_filepath(save_dir, ext=".wav"):
    """生成以时间命名的音频文件绝对路径 (例如: 20260917_093950.wav)，若存在同名则自动递增"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f"{timestamp}{ext}")
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(save_dir, f"{timestamp}_{counter}{ext}")
        counter += 1
    return filepath


def run_wake_word_recorder(args):
    if sd is None and not args.test_wav:
        print("[!] 错误: sounddevice 未正确安装或不可用。请检查声卡驱动及依赖。", file=sys.stderr)
        sys.exit(1)

    # 1. 初始化集成时空物理强约束的唤醒词检测器
    detector = KWSDetector(
        model_name=args.kws_model,
        checkpoint_path=args.kws_checkpoint,
        keywords=args.keywords,
        device=args.device,
        score_thresh=args.score_thresh,
        min_span=args.min_span,
        max_span=args.max_span,
        max_gap=args.max_gap,
        min_token_score=args.min_token_score,
        debug=args.debug
    )

    # 2. 可选初始化 ASR 识别模型
    asr_model = None
    if args.enable_asr:
        print(f"[*] 正在加载 ASR 识别模型: {args.asr_model} (设备: {args.device})...")
        asr_kwargs = {
            "model": args.asr_model,
            "device": args.device,
            "disable_update": True,
            "dtype": "fp16" if "cuda" in args.device else "fp32",
        }
        if args.vad_model:
            asr_kwargs["vad_model"] = args.vad_model
        asr_model = AutoModel(**asr_kwargs)
        print("[+] ASR 模型加载完成。")

    # 3. 初始化录音配置
    recorder = AudioStreamRecorder(
        sample_rate=16000,
        chunk_duration=0.1,
        window_duration=1.8,
        step_duration=0.1,
        silence_duration=args.silence_duration,
        silence_thresh=args.silence_thresh,
        save_dir=args.save_dir,
        mic_index=args.mic_index
    )

    # 4. 初始化键盘触发器（可选）
    keyboard_trigger = KeyboardTrigger(hotkey=args.hotkey, enabled=args.enable_hotkey)

    # 5. 如果是模拟测试模式（输入本地 wav 文件流）
    if args.test_wav:
        run_file_simulation(
            args.test_wav, detector, recorder, asr_model,
            debounce_hits=args.debounce_hits, safe_margin=args.safe_margin
        )
        return

    # 6. 实时麦克风录音循环
    if args.auto_calibrate:
        recorder.calibrate_noise(duration=1.0)

    keyboard_trigger.start()

    print("\n" + "=" * 65)
    print(" 唤醒词监听与自动录音系统已就绪 (时空物理双重防线) ".center(65, "="))
    print(f"  • 目标唤醒词   : {args.keywords}")
    print(f"  • 判定置信度   : {args.score_thresh} (单字最低分: {args.min_token_score})")
    print(f"  • 时空物理约束 : 跨度[{args.min_span}s~{args.max_span}s], 最大字间距<={args.max_gap}s, 防抖确认={args.debounce_hits}次")
    print(f"  • 尾音安全保护 : 保留 {args.safe_margin} 秒弱音余量 (双阈值迟滞 + 智能反向回溯，绝不截断尾音)")
    print(f"  • 手动触发按键 : {'按 ' + args.hotkey.upper() + ' 键立即录音' if args.enable_hotkey else '未启用'}")
    print(f"  • 结束录音条件 : 连续 {args.silence_duration} 秒无收音 (静音)")
    print(f"  • 录音保存目录 : {os.path.abspath(args.save_dir)}")
    print(f"  • ASR转写功能  : {'已启用' if args.enable_asr else '未启用'}")
    print("  • 按 Ctrl+C 可随时安全退出")
    print("=" * 65 + "\n")

    state = "LISTENING"  # "LISTENING" 或 "RECORDING"
    kws_buffer = np.zeros(0, dtype=np.float32)
    recorded_frames = []
    last_voice_time = 0.0
    accumulated_step_samples = 0
    record_start_time = 0.0
    record_source = "wakeword"
    consecutive_hits = 0

    print(f"👂 [正在监听] 请说出唤醒词 '{args.keywords}'...")

    with sd.InputStream(samplerate=recorder.sample_rate, channels=1, dtype="float32",
                        device=recorder.mic_index, callback=recorder._audio_callback,
                        blocksize=recorder.chunk_samples):
        try:
            while True:
                try:
                    chunk = recorder.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    if state == "LISTENING" and keyboard_trigger.consume():
                        print(f"\n⌨️  >>> [手动触发] 检测到 '{args.hotkey.upper()}' 键，立即开始录音！")
                        print("🔴 >>> [开始录音] 正在收音中，说话停顿满 3 秒将自动完成录音...")
                        state = "RECORDING"
                        recorded_frames = []
                        last_voice_time = time.time()
                        record_start_time = time.time()
                        kws_buffer = np.zeros(0, dtype=np.float32)
                        record_source = "manual"
                        consecutive_hits = 0
                    continue

                chunk_rms = np.sqrt(np.mean(chunk ** 2))

                if state == "LISTENING":
                    if keyboard_trigger.consume():
                        print(f"\n⌨️  >>> [手动触发] 检测到 '{args.hotkey.upper()}' 键，立即开始录音！")
                        print("🔴 >>> [开始录音] 正在收音中，说话停顿满 3 秒将自动完成录音...")
                        state = "RECORDING"
                        recorded_frames = []
                        last_voice_time = time.time()
                        record_start_time = time.time()
                        kws_buffer = np.zeros(0, dtype=np.float32)
                        record_source = "manual"
                        consecutive_hits = 0
                        continue

                    kws_buffer = np.concatenate([kws_buffer, chunk])
                    accumulated_step_samples += len(chunk)

                    if len(kws_buffer) > recorder.window_samples:
                        kws_buffer = kws_buffer[-recorder.window_samples:]

                    if (accumulated_step_samples >= recorder.step_samples and
                            len(kws_buffer) >= int(recorder.sample_rate * 0.8)):
                        accumulated_step_samples = 0
                        is_detected, kw, score = detector.detect(kws_buffer)
                        if is_detected:
                            consecutive_hits += 1
                            if consecutive_hits >= args.debounce_hits:
                                print(f"\n🎉 >>> [唤醒成功] 检测到唤醒词: '{kw}' (置信度: {score:.4f}, 防抖确认: {consecutive_hits}/{args.debounce_hits})")
                                print("🔴 >>> [开始录音] 正在收音中，说话停顿满 3 秒将自动完成录音...")
                                state = "RECORDING"
                                recorded_frames = []
                                last_voice_time = time.time()
                                record_start_time = time.time()
                                kws_buffer = np.zeros(0, dtype=np.float32)
                                record_source = "wakeword"
                                consecutive_hits = 0
                        else:
                            consecutive_hits = 0

                elif state == "RECORDING":
                    recorded_frames.append(chunk)
                    current_time = time.time()

                    if chunk_rms > recorder.silence_thresh:
                        last_voice_time = current_time
                        sys.stdout.write("\r🎙️  正在收音中... [音量: {:.4f}]".format(chunk_rms))
                        sys.stdout.flush()
                    else:
                        silence_elapsed = current_time - last_voice_time
                        sys.stdout.write(f"\r⏳ 静音检测中: 已静音 {silence_elapsed:.1f}s / {recorder.silence_duration}s [音量: {chunk_rms:.4f}]")
                        sys.stdout.flush()

                        if silence_elapsed >= recorder.silence_duration:
                            print("\n\n⏹️  >>> [录音结束] 连续 3 秒没有收音，自动停止录音。")
                            total_audio = np.concatenate(recorded_frames)

                            # 智能反向回溯定位真实语音终点，并保留 safe_margin (默认 0.8s) 安全尾音
                            save_audio = trim_trailing_silence(
                                total_audio,
                                sample_rate=recorder.sample_rate,
                                silence_thresh=recorder.silence_thresh,
                                safe_margin=args.safe_margin
                            )

                            save_filepath = generate_timestamp_filepath(recorder.save_dir, ext=".wav")
                            sf.write(save_filepath, save_audio, recorder.sample_rate)

                            duration_sec = len(save_audio) / recorder.sample_rate
                            print(f"💾 录音已保存至: {save_filepath} (有效时长: {duration_sec:.2f} 秒，含 {args.safe_margin}s 安全尾音缓冲)")

                            if asr_model is not None:
                                print("🤖 正在进行语音识别转写...")
                                asr_res = asr_model.generate(input=save_filepath, cache={}, batch_size_s=0)
                                if asr_res and len(asr_res) > 0:
                                    text = asr_res[0].get("text", "")
                                    print(f"📝 识别结果: {text}")
                                else:
                                    print("📝 识别结果为空。")

                            state = "LISTENING"
                            kws_buffer = np.zeros(0, dtype=np.float32)
                            recorded_frames = []
                            accumulated_step_samples = 0
                            record_source = "wakeword"
                            consecutive_hits = 0
                            print("\n" + "-" * 50)
                            print(f"👂 [重新监听] 请说出唤醒词 '{args.keywords}'...")

        except KeyboardInterrupt:
            keyboard_trigger.stop()
            print("\n\n[👋] 用户手动中断，程序已安全退出。")


def run_file_simulation(wav_path, detector, recorder, asr_model=None, debounce_hits=1, safe_margin=0.8):
    """用于无麦克风环境下的音频文件仿真推流测试"""
    print(f"[*] 启动音频文件仿真流测试: {wav_path}")
    try:
        import torchaudio
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != recorder.sample_rate:
            print(f"[*] 正在将音频重采样: {sr}Hz -> {recorder.sample_rate}Hz...")
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=recorder.sample_rate)
            waveform = resampler(waveform)
        data = waveform.squeeze().numpy().astype(np.float32)
    except Exception as e:
        data, sr = sf.read(wav_path, dtype="float32")
        if len(data.shape) > 1:
            data = data[:, 0]
        if sr != recorder.sample_rate:
            print(f"[!] 警告: 测试音频采样率为 {sr}Hz，建议转为 16000Hz 进行测试。")

    state = "LISTENING"
    kws_buffer = np.zeros(0, dtype=np.float32)
    recorded_frames = []
    last_voice_time = 0.0
    accumulated_step_samples = 0
    sim_time = 0.0
    wake_count = 0
    consecutive_hits = 0

    chunk_size = recorder.chunk_samples
    num_chunks = int(len(data) // chunk_size) + 1

    for i in range(num_chunks):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) == 0:
            break
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        sim_time += len(chunk) / recorder.sample_rate
        chunk_rms = np.sqrt(np.mean(chunk ** 2))

        if state == "LISTENING":
            kws_buffer = np.concatenate([kws_buffer, chunk])
            accumulated_step_samples += len(chunk)
            if len(kws_buffer) > recorder.window_samples:
                kws_buffer = kws_buffer[-recorder.window_samples:]

            if (accumulated_step_samples >= recorder.step_samples and
                    len(kws_buffer) >= int(recorder.sample_rate * 0.8)):
                accumulated_step_samples = 0
                is_detected, kw, score = detector.detect(kws_buffer)
                if is_detected:
                    consecutive_hits += 1
                    if consecutive_hits >= debounce_hits:
                        wake_count += 1
                        print(f"\n[{sim_time:.2f}s] 🎉 >>> 第 {wake_count} 次唤醒成功！检测到唤醒词: '{kw}' (置信度: {score:.4f}, 防抖: {consecutive_hits}/{debounce_hits})")
                        print(f"[{sim_time:.2f}s] 🔴 >>> 启动录音功能！正在收音中...")
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
                silence_elapsed = sim_time - last_voice_time
                if silence_elapsed >= recorder.silence_duration:
                    print(f"\n[{sim_time:.2f}s] ⏹️  >>> 检测到连续 {recorder.silence_duration} 秒没有收音，自动结束第 {wake_count} 次录音！")
                    total_audio = np.concatenate(recorded_frames)
                    save_audio = trim_trailing_silence(
                        total_audio,
                        sample_rate=recorder.sample_rate,
                        silence_thresh=recorder.silence_thresh,
                        safe_margin=safe_margin
                    )
                    save_filepath = generate_timestamp_filepath(recorder.save_dir, ext=".wav")
                    sf.write(save_filepath, save_audio, recorder.sample_rate)
                    print(f"💾 录音已保存至: {save_filepath} (有效时长: {len(save_audio)/recorder.sample_rate:.2f}s，含 {safe_margin}s 尾音保护余量)")
                    if asr_model:
                        asr_res = asr_model.generate(input=save_filepath, disable_pbar=True)
                        print(f"📝 ASR 识别结果: {asr_res[0].get('text', '')}")
                    state = "LISTENING"
                    kws_buffer = np.zeros(0, dtype=np.float32)
                    recorded_frames = []
                    consecutive_hits = 0
                    print(f"👂 重新进入监听状态，等待下一次唤醒...\n")

    if state == "RECORDING" and len(recorded_frames) > 0:
        print(f"\n[{sim_time:.2f}s] ℹ️  >>> 音频流已播放完毕，自动结算并保存本次录音...")
        total_audio = np.concatenate(recorded_frames)
        save_audio = trim_trailing_silence(
            total_audio,
            sample_rate=recorder.sample_rate,
            silence_thresh=recorder.silence_thresh,
            safe_margin=safe_margin
        )
        save_filepath = generate_timestamp_filepath(recorder.save_dir, ext=".wav")
        sf.write(save_filepath, save_audio, recorder.sample_rate)
        print(f"💾 录音已保存至: {save_filepath} (时长: {len(save_audio)/recorder.sample_rate:.2f}s)")
        if asr_model:
            asr_res = asr_model.generate(input=save_filepath)
            print(f"📝 ASR 识别结果: {asr_res[0].get('text', '')}")


def parse_args():
    parser = argparse.ArgumentParser(description="FunASR 唤醒词触发录音脚本 (时空物理约束双重防线)")
    parser.add_argument("--kws_model", type=str,
                        default="/home/inno/.cache/modelscope/hub/models/iic/speech_sanm_kws_phone-xiaoyun-commands-online",
                        help="唤醒词底模名称或本地路径")
    parser.add_argument("--kws_checkpoint", type=str,
                        default="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3/model.pt.best",
                        help="微调后的 checkpoint 路径 (例如 model.pt.best)")
    parser.add_argument("--keywords", type=str, default="鹰眼鹰眼",
                        help="目标唤醒词，多个可用逗号隔开")
    parser.add_argument("--score_thresh", type=float, default=0.8,
                        help="综合置信度阈值 (默认: 0.8)")
    parser.add_argument("--min_token_score", type=float, default=0.65,
                        help="每个音节独立最低置信度门限 (默认: 0.65)")
    parser.add_argument("--min_span", type=float, default=0.40,
                        help="4字最短发音总时长秒数 (默认: 0.40s)")
    parser.add_argument("--max_span", type=float, default=1.35,
                        help="4字最长发音总时长秒数 (默认: 1.35s)")
    parser.add_argument("--max_gap", type=float, default=0.50,
                        help="相邻音节间最大允许停顿秒数 (默认: 0.50s)")
    parser.add_argument("--debounce_hits", type=int, default=1,
                        help="防抖连续命中次数确认 (默认: 1，极嘈杂环境可设为 2)")
    parser.add_argument("--silence_duration", type=float, default=3.0,
                        help="录音结束标志: 无收音 (静音) 持续秒数 (默认: 3.0)")
    parser.add_argument("--safe_margin", type=float, default=0.8,
                        help="真实语音截止后保留的尾音安全余量秒数 (默认: 0.8s)")
    parser.add_argument("--silence_thresh", type=float, default=0.015,
                        help="静音判断 RMS 能量阈值 (默认: 0.015)")
    parser.add_argument("--auto_calibrate", action="store_true",
                        help="启动时自动采集 1 秒环境噪音并自适应设定静音阈值")
    parser.add_argument("--save_dir", type=str, default="/media/inno/output/ASR/唤醒词/v3/",
                        help="录音保存目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备 (默认: cuda:0，无GPU时可设为 cpu)")
    parser.add_argument("--mic_index", type=int, default=None,
                        help="音频输入设备索引 (不指定则使用系统默认输入设备)")
    parser.add_argument("--enable_asr", action="store_true",
                        help="录音结束后是否自动进行 ASR 语音识别转写")
    parser.add_argument("--asr_model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512",
                        help="ASR 模型名称或本地路径")
    parser.add_argument("--vad_model", type=str, default=None,
                        help="VAD 模型名称 (例如 fsmn-vad，默认 None)")
    # parser.add_argument("--test_wav", type=str, default="/media/inno/ASR/KWS/误唤醒分析/output/wake_words/wake_20260923_181900_sc0_884.wav",
    parser.add_argument("--test_wav", type=str, default="/media/inno/ASR/KWS/test/唤醒词-鹰眼鹰眼.wav",
                        help="测试模式：传入 wav 文件路径模拟实时推流，用于验证")
    parser.add_argument("--enable_hotkey", action="store_true",
                        help="启用按热键手动触发录音（默认关闭）")
    parser.add_argument("--hotkey", type=str, default="s",
                        help="手动触发录音的按键（默认: s）")
    parser.add_argument("--debug", action="store_true",
                        help="是否打印时空物理约束诊断日志")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_wake_word_recorder(args)