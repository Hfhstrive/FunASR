import time
from funasr import AutoModel

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
# finutune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v2/"
finutune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/"

# 热词权值以:隔开
with open('/media/inno/ASR/gi_hotwords_v3.txt', "r", encoding="utf-8") as f:
    hotword_list = [line.strip() for line in f if line.strip()]
model = AutoModel(
    model=finutune_checkpoint,
    # model=model_dir,
    # vad_model="fsmn-vad",
    device="cuda:0",
    hotwords=hotword_list,
    # trust_remote_code=True,
    disable_update=True,
    dtype="fp16",
)
wav_paths = [
    '/media/inno/ASR/input/test2.wav',
    '/media/inno/ASR/audio/test/张学松.mp3',
    '/media/inno/ASR/audio/test/0601/test.mp3',
    '/media/inno/ASR/audio/test/hfh/胃癌.mp3',
    '/media/inno/ASR/audio/test/hfh/反流性食管炎.mp3',
    '/media/inno/ASR/audio/test/hfh/萎缩肠化.mp3',
]
for wav_path in wav_paths:
    res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
    text = res[0]["text"]
    print(text)


# -------------------------- 唤醒词 + ASR -------------------------------
# from funasr import AutoModel
# model = AutoModel(model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
#                   keywords="你好小云",
#                   output_dir="./outputs/debug",
#                   device='cpu',
#                   chunk_size=[4, 8, 4],
#                   encoder_chunk_look_back=0,
#                   decoder_chunk_look_back=0,
#                  )
#
# res = model.generate(input=f"/media/inno/ASR/input/test2.wav")
# print(res)


# import re
# from funasr import AutoModel
#
# # 1. 初始化 KWS 模型
# kws_model = AutoModel(
#     model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
#     keywords="你好小云",
#     output_dir="./outputs/debug",
#     device='cuda:0',
#     chunk_size=[4, 8, 4],
#     encoder_chunk_look_back=0,
#     decoder_chunk_look_back=0,
# )
#
# # 2. 初始化 ASR 模型（利用其自带的 VAD 进行长音频切分）
# asr_model = AutoModel(
#     model="FunAudioLLM/Fun-ASR-Nano-2512",
#     vad_model="fsmn-vad",
#     device="cuda:0",
#     # hotwords=['斑驳', '发红'],
#     hotwords='/media/inno/ASR/gi_hotwords.txt',
#     trust_remote_code=True
# )
#
# wav_path = "/media/inno/ASR/input/test1.wav"
#
# # --- 推理与解析逻辑 ---
#
# # 执行 KWS 推理
# kws_res = kws_model.generate(input=wav_path)
# print(f"KWS 原始输出: {kws_res}")
#
# # 解析格式: [{'key': 'test2', 'text': 'detected 你好小云 0.8437...'}]
# if len(kws_res) > 0:
#     res_text = kws_res[0].get('text', "")
#
#     # 使用正则表达式提取检测状态和置信度
#     match = re.search(r"detected\s+(\S+)\s+([\d\.]+)", res_text)
#
#     if match:
#         keyword = match.group(1)
#         score = float(match.group(2))
#
#         # 设定唤醒阈值，噪声环境下建议 0.6 以上
#         if score > 0.5:
#             print(f">>> 唤醒成功！检测到：{keyword} (置信度: {score:.4f})")
#
#             # --- 难点：由于没有 offset，我们采用 VAD 自动切分识别方案 ---
#             # 直接将全文交给带 VAD 的 ASR 模型
#             # ASR 会自动识别出所有片段，我们只需要过滤掉唤醒词之前的内容
#
#             asr_res = asr_model.generate(
#                 input=wav_path,
#                 cache={},
#                 batch_size_s=0
#             )
#
#             if len(asr_res) > 0:
#                 full_transcription = asr_res[0]['text']
#                 print(f"ASR 原始文本: {full_transcription}")
#                 # 1. 定义需要过滤的唤醒词正则表达式
#                 # [，。？！、\s]* 表示匹配 0 个或多个常见的标点符号或空格
#                 # 这样可以匹配： "你好小云", "你好，小云", "你好小，云", "你好 小云" 等
#                 pattern = r"你好[，。？！、\s]*小[，。？！、\s]*云"
#                 # 2. 使用正则进行拆分
#                 # maxsplit=1 表示只拆分第一次出现的唤醒词，后面的内容全部视为指令
#                 parts = re.split(pattern, full_transcription, maxsplit=1)
#                 if len(parts) > 1:
#                     # 取得拆分后的最后一部分即为指令
#                     command = parts[1].strip()
#                     # 3. 清理指令开头的残余标点
#                     command = re.sub(r"^[，。？！、\s]+", "", command)
#                     print(f"【精准指令识别】: {command}")
#                 else:
#                     # 如果没匹配到正则（可能是识别成了完全不同的字），则输出全文
#                     print(f"【未匹配唤醒词，全句输出】: {full_transcription}")
#         else:
#             print(f"检测到疑似唤醒，但置信度过低 ({score})")
#     else:
#         print("未检测到唤醒关键词。")
