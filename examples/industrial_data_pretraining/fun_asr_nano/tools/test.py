import time
import os
import sys
import torch
import psutil
import glob
import json
from funasr import AutoModel

# 初始化 psutil 采样器基准线
psutil.cpu_percent(interval=None)
psutil.Process(os.getpid()).cpu_percent(interval=None)

def get_resource_usage(device="cuda:0"):
    usage = {}
    usage["cpu_percent"] = psutil.cpu_percent(interval=None)
    process = psutil.Process(os.getpid())
    usage["process_cpu_percent"] = process.cpu_percent(interval=None)
    
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            used_mem = total_mem - free_mem
            usage["gpu_total_mb"] = total_mem / 1024 / 1024
            usage["gpu_used_mb"] = used_mem / 1024 / 1024
            usage["gpu_percent"] = (used_mem / total_mem) * 100
            
            torch_allocated = torch.cuda.memory_allocated(device)
            usage["torch_allocated_mb"] = torch_allocated / 1024 / 1024
            usage["torch_percent"] = (torch_allocated / total_mem) * 100
        except Exception as e:
            usage["gpu_error"] = str(e)
    return usage

# 动态引入 asr-hotword 项目，并提供环境依赖诊断
sys.path.append('/home/inno/code/ASR/asr-hotword')
from hotword import PhonemeCorrector

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
# finutune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v2/"
# finutune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/"
finutune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/"

# 1. 纯净状态初始化 ASR 模型（不加载 hotwords 参数，避免干扰 LLM 解码）
print("正在初始化 ASR 模型并计时...")
init_start_time = time.time()
model = AutoModel(
    model=finutune_checkpoint,
    # init_param=os.path.join(finutune_checkpoint, "model.pt.avg3"), 
    # model=model_dir,
    # vad_model="fsmn-vad",
    device="cuda:0",
    # trust_remote_code=True,
    disable_update=True,
    dtype="fp16",
)
init_end_time = time.time()
init_time = init_end_time - init_start_time

usage_after_init = get_resource_usage(device="cuda:0")
print(f"ASR 模型初始化耗时: {init_time:.4f} 秒")
if "gpu_percent" in usage_after_init:
    print(f"初始化后显存占比 (系统整体): {usage_after_init['gpu_percent']:.2f}% ({usage_after_init['gpu_used_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")
    print(f"初始化后显存占比 (PyTorch分配): {usage_after_init['torch_percent']:.2f}% ({usage_after_init['torch_allocated_mb']:.1f}MB / {usage_after_init['gpu_total_mb']:.1f}MB)")


# 2. 初始化音素级后处理纠错器
print("正在加载后处理医学纠错词表...")
hotword_txt_path = '/media/inno/ASR/gi_hotwords.txt'
corrector = PhonemeCorrector(threshold=0.85)
if os.path.exists(hotword_txt_path):
    with open(hotword_txt_path, "r", encoding="utf-8") as f:
        hotwords_content = f.read()
    corrector.update_hotwords(hotwords_content)
    print(f"纠错词表加载完成。")
else:
    print(f"警告：未找到热词文件 {hotword_txt_path}")

audio_paths = [
    '/media/inno/ASR/胃镜/input/test2.wav',
    '/media/inno/ASR/胃镜/audio/test/zxs/test.mp3',
    '/media/inno/ASR/胃镜/audio/test/0601/test.mp3',
    '/media/inno/ASR/胃镜/audio/test/hfh/胃癌.mp3',
    '/media/inno/ASR/胃镜/audio/test/hfh/反流性食管炎.mp3',
    '/media/inno/ASR/胃镜/audio/test/hfh/萎缩肠化.mp3',
]
val_path = '/media/inno/ASR/ChatML/V4/val'
audio_paths = glob.glob(f'{val_path}/src0/*.wav') + glob.glob(f'{val_path}/src1/*.wav')

# 结果保存配置
result_save_dir = '/media/inno/output/ASR/ChatML/V4/val/'
os.makedirs(result_save_dir, exist_ok=True)
results_json_path = os.path.join(result_save_dir, "val_results.json")
results_dict = {}

# 重置 psutil 计时基准
psutil.cpu_percent(interval=None)
psutil.Process(os.getpid()).cpu_percent(interval=None)
total_infer_start = time.time()

for audio_path in audio_paths:
    audio_type = audio_path.split('.')[-1]
    if audio_type != 'wav':
        wav_path = audio_path.replace(audio_type, 'wav')
        if not os.path.exists(wav_path):
            os.system(f'ffmpeg -i {audio_path} -acodec pcm_s16le -ar 16000 -ac 1 -y {wav_path}')
    else:
        wav_path = audio_path
        
    # ASR 模型推理得到原始文本
    infer_single_start = time.time()
    res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
    raw_text = res[0]["text"]
    
    # 执行音素级纠错后处理
    corrected_res = corrector.correct(raw_text)
    final_text = corrected_res.text
    infer_single_time = time.time() - infer_single_start
    
    print("-" * 50)
    print(f"音频: {os.path.basename(audio_path)} (推理+纠错单句耗时: {infer_single_time:.4f} 秒)")
    print(f"原始识别: {raw_text}")
    print(f"纠错结果: {final_text}")
    if corrected_res.matches:
        print(f"纠错替换: {corrected_res.matches}")
        
    # 提取唯一的 case 键 (例如 src0_case-no) 并存入字典
    parts = audio_path.split('/')
    src_name = parts[-2] if len(parts) > 1 else "src0"
    case_name = os.path.basename(audio_path).replace('.wav', '')
    case_key = f"{src_name}_{case_name}"
    
    results_dict[case_key] = {
        "asr_result": raw_text,
        "corrected_result": final_text
    }

total_infer_time = time.time() - total_infer_start

# 获取推理结束时的资源占用
usage_after_infer = get_resource_usage(device="cuda:0")
# 获取推理期间的 CPU 占比（在此时间段内的平均值）
cpu_infer_system = psutil.cpu_percent(interval=None)
cpu_infer_process = psutil.Process(os.getpid()).cpu_percent(interval=None)
cpu_count = psutil.cpu_count() or 1
process_cpu_ratio = cpu_infer_process / cpu_count

print("\n" + "=" * 60)
print(" 性能统计结果 ".center(60, "="))
print(f"- 模型初始化时间: {init_time:.4f} 秒")
print(f"- 总推理运行时间: {total_infer_time:.4f} 秒")
if "gpu_percent" in usage_after_infer:
    print(f"- 推理后显存占比 (系统整体): {usage_after_infer['gpu_percent']:.2f}% ({usage_after_infer['gpu_used_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
    print(f"- 推理后显存占比 (PyTorch分配): {usage_after_infer['torch_percent']:.2f}% ({usage_after_infer['torch_allocated_mb']:.1f}MB / {usage_after_infer['gpu_total_mb']:.1f}MB)")
print(f"- 进程 CPU 占比: {cpu_infer_process:.2f}% (折算系统整体占用为: {process_cpu_ratio:.2f}%)")
print(f"- 系统总 CPU 占比: {cpu_infer_system:.2f}%")
# 统一保存为 json 文件
with open(results_json_path, "w", encoding="utf-8") as fj:
    json.dump(results_dict, fj, ensure_ascii=False, indent=4)

print(f"- 推理结果已统一保存至 JSON: {results_json_path}")
print("=" * 60)



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
