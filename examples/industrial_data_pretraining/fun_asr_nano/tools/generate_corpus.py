import glob
import os
import random
import time
import websocket
import datetime
import hashlib
import hmac
import base64
import json
import ssl
from urllib.parse import urlencode
import shutil

# --- 讯飞 TTS 配置区域 ---
APP_ID = '9d79363d'
API_KEY = '69b0c6ffca5e506c0a3625d8754e87ca'
API_SECRET = 'NjA1NWI5NDUzOGNiYTA4NWExMDA3NWMz'

# 发音人列表
VCN_LIST = [
    # "x4_yezi", # 讯飞小露
    # "x4_lingbosong", # 聆伯松
    # "x4_chaoge", # 讯飞超哥
    # "x4_lingxiaoxuan_en_v2", # 聆小璇-助理
    # "x2_wanshu", # 讯飞万叔
    # "x4_lingfeizhe_zl",  # 聆飞哲
    "xiaoyan",   # 小燕（经典女声）
    "xiaofeng",  # 小峰（经典男声）
    "aisjiuxu",  # 久馨（标准女声）
    "aisxping"   # 新平（标准男声）
]

class TTS_Client:
    def __init__(self, app_id, api_key, api_secret, text, save_path):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.text = text
        self.save_path = save_path
        self.host = "tts-api.xfyun.cn"

        # 随机参数生成以增加语音多样性
        self.vcn = random.choice(VCN_LIST)
        self.speed = random.randint(40, 55)
        self.volume = random.randint(40, 80)

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.datetime.now()
        date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        signature_origin = f"host: {self.host}\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", ' \
                               f'headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {"authorization": authorization, "date": date, "host": self.host}
        return url + '?' + urlencode(v)

    def on_message(self, ws, message):
        try:
            res = json.loads(message)
            code = res["code"]
            if code != 0:
                print(f"Error: {res['message']}")
                return

            audio = base64.b64decode(res["data"]["audio"])
            status = res["data"]["status"]
            with open(self.save_path, 'ab') as f:
                f.write(audio)

            if status == 2:
                ws.close()
        except Exception as e:
            print("receive msg error:", e)

    def on_open(self, ws):
        def run(*args):
            d = {
                "common": {"app_id": self.app_id},
                "business": {
                    "aue": "lame",
                    "sfl": 1,
                    "vcn": self.vcn,
                    "speed": self.speed,
                    "volume": self.volume,
                    "tte": "UTF8"
                },
                "data": {"status": 2, "text": str(base64.b64encode(self.text.encode('utf-8')), "UTF-8")}
            }
            ws.send(json.dumps(d))

        import _thread
        _thread.start_new_thread(run, ())

    def start(self):
        ws_url = self.create_url()
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

        print(f">>> 启动 TTS 合成 | 发音人: {self.vcn}, 语速: {self.speed}, 音量: {self.volume}")
        ws = websocket.WebSocketApp(ws_url, on_message=self.on_message,
                                    on_error=lambda w, e: print(e),
                                    on_close=lambda w, a, b: None)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


def generate_wav_with_retry(word_text, target_wav_path, max_retry=5, base_sleep=1.0):
    if os.path.exists(target_wav_path):
        return

    for retry_idx in range(max_retry):
        temp_mp3 = target_wav_path.replace('.wav', '.mp3')
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        try:
            client = TTS_Client(APP_ID, API_KEY, API_SECRET, word_text, temp_mp3)
            client.start()

            if os.path.exists(temp_mp3):
                cmd = f"ffmpeg -y -i {temp_mp3} -acodec pcm_s16le -ar 16000 -ac 1 {target_wav_path} > /dev/null 2>&1"
                os.system(cmd)
                os.remove(temp_mp3)
        except Exception as e:
            print(f"[TTS] 捕获到 SSL 或连接异常: {e}")

        if os.path.exists(target_wav_path):
            time.sleep(base_sleep)
            return

        # 异常或合成失败，采取指数退避重试，避开限流
        wait_time = base_sleep * (2 ** retry_idx)
        print(f"[TTS] 音频生成失败或连接被切断，{wait_time} 秒后重试 (第 {retry_idx + 1}/{max_retry} 次)...")
        time.sleep(wait_time)

    raise RuntimeError(f"WAV 音频合成转码失败，已重试 {max_retry} 次: {target_wav_path}")


def corpus_convert(word_path, speech_path, save_path, split_lines=False, source_idx=0, case_mapping=None, case_modes=None):
    if case_mapping is None:
        case_mapping = {}
    if case_modes is None:
        case_modes = {}

    # 遍历word_path 下的文件夹，根据文件夹划分数据集
    for lesion in os.listdir(word_path):
        train_nums, val_nums = 0, 0
        work_lesion_path = os.path.join(word_path, lesion)
        if not os.path.isdir(work_lesion_path):
            continue
            
        lesion_words = glob.glob(f'{work_lesion_path}/**.txt')
        for lesion_word in lesion_words:
            # 原始病例的文件名
            original_case_no = lesion_word.split('/')[-1].split('.')[0]
            # 为了全局唯一性且保留数据源标识，加上前缀（例如 src0_583_3516）
            case_no = f"src{source_idx}_{original_case_no}"
            
            with open(lesion_word, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            clean_lines = [line.strip('\n').strip().rstrip('。') for line in lines if line.strip('\n').strip()]
            assert len(clean_lines) > 0, f"文件内容为空: {lesion_word}"
            
            # 优先从全局 case_modes 中读取已分配的 mode，保证跨数据源时相同 case_no 的样本被分配到同一集合 (train/val)，避免泄露
            if original_case_no in case_modes:
                mode = case_modes[original_case_no]
                if mode == 'train':
                    train_nums += 1
                else:
                    val_nums += 1
            else:
                # 首次出现，按照设定的比例策略划分
                if val_nums >= len(lesion_words) * 0.1:
                    mode = 'train'
                    train_nums += 1
                elif val_nums == 0 or random.random() <= 0.1:
                    mode = 'val'
                    val_nums += 1
                else:
                    mode = 'train'
                    train_nums += 1
                # 记录该病例的划分
                case_modes[original_case_no] = mode
                
            save_scp_path = os.path.join(save_path, mode + '.scp')
            save_txt_path = os.path.join(save_path, mode + '.txt')

            if split_lines and len(clean_lines) > 1:
                # 开启多行切分，且有至少两行内容
                for i, line_content in enumerate(clean_lines):
                    sub_case_no = f"{case_no}_{i}"
                    word_info = line_content + '。'
                    lesion_speech = lesion_word.replace(word_path, speech_path).replace('.txt', f'_{i}.wav')
                    
                    if not os.path.exists(lesion_speech):
                        os.makedirs(os.path.dirname(lesion_speech), exist_ok=True)
                        print(f"\n[TTS Sub] 子音频不存在，正在生成: '{word_info}'")
                        generate_wav_with_retry(word_info, lesion_speech)
                        print(f"[TTS Sub] 成功生成子音频 -> {lesion_speech}")
                        
                    with open(save_scp_path, 'a+', encoding='utf-8') as f1:
                        f1.writelines(sub_case_no + ' ' + lesion_speech + '\n')
                    with open(save_txt_path, 'a+', encoding='utf-8') as f1:
                        f1.writelines(sub_case_no + ' ' + word_info + '\n')
                        
                    if mode == 'val':
                        sub_save_dir = os.path.join(save_path, 'val', f"src{source_idx}")
                        os.makedirs(sub_save_dir, exist_ok=True)
                        val_save_wav = os.path.join(sub_save_dir, os.path.basename(lesion_speech))
                        try:
                            shutil.copy2(lesion_speech, val_save_wav)
                        except Exception as e:
                            print(f"[Copy Val] 拷贝失败: {e}")
                            
                    # 记录映射关系，lesion 是病例文件夹名称（即病例号）
                    case_mapping[sub_case_no] = {
                        "case_id": lesion,
                        "original_name": original_case_no,
                        "source_index": source_idx,
                        "split_index": i,
                        "dataset": mode
                    }
            else:
                # 未开启多行切分，或仅有一行
                word_info = '。'.join(clean_lines) + '。'
                lesion_speech = lesion_word.replace(word_path, speech_path).replace('.txt', '.wav')
                
                # 检测音频文件是否存在，若不存在则调用讯飞 TTS 合成并转码
                if not os.path.exists(lesion_speech):
                    os.makedirs(os.path.dirname(lesion_speech), exist_ok=True)
                    print(f"\n[TTS] 音频不存在，正在生成: '{word_info}'")
                    generate_wav_with_retry(word_info, lesion_speech)
                    print(f"[TTS] 成功生成 WAV 并对齐 16k mono 格式 -> {lesion_speech}")
                    
                with open(save_scp_path, 'a+', encoding='utf-8') as f1:
                    f1.writelines(case_no + ' ' + lesion_speech + '\n')
                with open(save_txt_path, 'a+', encoding='utf-8') as f1:
                    f1.writelines(case_no + ' ' + word_info + '\n')
                    
                # if mode == 'val':
                sub_save_dir = os.path.join(save_path, mode, f"src{source_idx}")
                os.makedirs(sub_save_dir, exist_ok=True)
                val_save_wav = os.path.join(sub_save_dir, os.path.basename(lesion_speech))
                try:
                    shutil.copy2(lesion_speech, val_save_wav)
                except Exception as e:
                    print(f"[Copy Val] 拷贝失败: {e}")
                        
                # 记录映射关系，lesion 是病例文件夹名称（即病例号）
                case_mapping[case_no] = {
                    "case_id": lesion,
                    "original_name": original_case_no,
                    "source_index": source_idx,
                    "split_index": None,
                    "dataset": mode
                }


if __name__ == '__main__':
    random.seed(20260528)
    save_path = '/media/inno/ASR/ChatML/V5/'
    os.makedirs(save_path, exist_ok=True)

    # # 清理已存在的输出文件以防重复追加
    # for mode in ['train', 'val']:
    #     for ext in ['.scp', '.txt']:
    #         file_to_remove = os.path.join(save_path, f'{mode}{ext}')
    #         if os.path.exists(file_to_remove):
    #             os.remove(file_to_remove)

    # 输入数据源配置列表 (word_path, speech_path, split_lines)
    data_sources = [
        ('/media/inno/ASR/胃镜/base_data/oral/case/', '/media/inno/ASR/胃镜/audio/train/real/case/', False),
        ('/media/inno/ASR/肠镜/base_data/oral/case/', '/media/inno/ASR/肠镜/audio/train/real/case/', False),
        # ('/media/inno/ASR/胃镜/base_data/standard/case/', '/media/inno/ASR/胃镜/audio/train/AIkit/standard/case/', True),
        # ('/media/inno/ASR/肠镜/base_data/standard_1/case/', '/media/inno/ASR/肠镜/audio/train/AIkit/standard_1/case/', True)
    ]

    case_mapping = {}
    case_modes = {}  # 全局记录每个病例文件名 (case_no) 的划分集合，确保跨数据源时保持一致

    # 依次处理数据源并写入保存路径
    for idx, (word_path, speech_path, split_lines) in enumerate(data_sources):
        corpus_convert(word_path, speech_path, save_path, split_lines=split_lines, source_idx=idx, case_mapping=case_mapping, case_modes=case_modes)

    # 提取 train 和 val 下的不重复原始病例名 (original_name) 以及对应的 ASR ID 列表
    train_cases = sorted(list(set(v["original_name"] for v in case_mapping.values() if v["dataset"] == "train")))
    val_cases = sorted(list(set(v["original_name"] for v in case_mapping.values() if v["dataset"] == "val")))
    train_asr_ids = sorted([k for k, v in case_mapping.items() if v["dataset"] == "train"])
    val_asr_ids = sorted([k for k, v in case_mapping.items() if v["dataset"] == "val"])

    output_data = {
        "train_cases": train_cases,       # 训练集包含的原始病例文件名列表（已去重）
        "val_cases": val_cases,           # 验证集包含的原始病例文件名列表（已去重）
        "train_asr_ids": train_asr_ids,   # 训练集对应的 ASR 最终唯一识别号列表
        "val_asr_ids": val_asr_ids,       # 验证集对应的 ASR 最终唯一识别号列表
        "mappings": case_mapping          # 详细映射字典（Key 为 ASR 识别号）
    }

    # 将映射关系与汇总列表保存到 json
    mapping_json_path = os.path.join(save_path, 'case_mapping.json')
    with open(mapping_json_path, 'w', encoding='utf-8') as fj:
        json.dump(output_data, fj, ensure_ascii=False, indent=4)
    print(f"\n>>> 数据集映射与汇总已成功保存至 {mapping_json_path}")