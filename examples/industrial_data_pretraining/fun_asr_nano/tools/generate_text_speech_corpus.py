import glob
import os
import websocket
import random
import datetime
import hashlib
import hmac
import base64
import json
import time
import ssl
from urllib.parse import urlencode
from ipdb import set_trace


class TTS_Client:
    def __init__(self, app_id, api_key, api_secret, VCN_LIST, text, save_path):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.text = text
        self.save_path = save_path
        self.host = "tts-api.xfyun.cn"

        # --- 随机参数生成 ---
        self.vcn = random.choice(VCN_LIST)  # 随机发音人
        self.speed = random.randint(40, 55)  # 随机语速 0-100，通常40-70较自然
        self.volume = random.randint(40, 80)  # 随机音量 0-100
        # ------------------

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
            # 将随机参数放入 business 字段
            d = {
                "common": {"app_id": self.app_id},
                "business": {
                    "aue": "lame",
                    "sfl": 1,
                    "vcn": self.vcn,  # 随机发音人
                    "speed": self.speed,  # 随机语速
                    "volume": self.volume,  # 随机音量
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

        print(f">>> 发音人: {self.vcn}, 语速: {self.speed}, 音量: {self.volume}")
        ws = websocket.WebSocketApp(ws_url, on_message=self.on_message,
                                    on_error=lambda w, e: print(e),
                                    on_close=lambda w, a, b: None)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


if __name__ == '__main__':
    # 语音配置
    APP_ID = '9d79363d'
    API_KEY = '69b0c6ffca5e506c0a3625d8754e87ca'
    API_SECRET = 'NjA1NWI5NDUzOGNiYTA4NWExMDA3NWMz'
    # 备选发音人列表 (可以根据你的权限在讯飞控制台添加更多)
    VCN_LIST = [
        "x4_yezi",  # 讯飞小露
        "x4_lingbosong",  # 聆伯松
        "x4_chaoge",  # 讯飞超哥
        "x4_lingxiaoxuan_en_v2",  # 聆小璇-助理
        "x2_wanshu",  # 讯飞万叔
        "x4_lingfeizhe_zl",  # 聆飞哲
    ]

    txt_dir = '/media/inno/ASR/base_data/oral/'
    audio_dir = '/media/inno/ASR/output_audio/oral/'
    save_dir = '/media/inno/ASR/ChatML/'

    txt_paths = glob.glob(f'{txt_dir}/**.txt')
    for txt_path in txt_paths:
        name = txt_path.split('/')[-1].split('.')[0]
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                Utterance_ID = name + '_line' + str(idx)
                speech = line.strip('\n')
                audio_path = os.path.join(audio_dir, f'{Utterance_ID}.mp3')
                # 生成语音文件
                print(f"处理文件: {name} 第 {idx} 行...")
                client = TTS_Client(APP_ID, API_KEY, API_SECRET, VCN_LIST, speech, audio_path)
                client.start()
                time.sleep(0.5)
                # assert os.path.exists(audio_path)
                # 写入文档
                mode = 'train' if idx <= 0.8 * len(lines) else 'val'
                save_text_path = os.path.join(save_dir, f'{mode}.txt')
                save_audio_path = os.path.join(save_dir, f'{mode}.scp')
                with open(save_text_path, 'a+') as f1:
                    f1.writelines(Utterance_ID + ' ' + speech + '\n')
                with open(save_audio_path, 'a+') as f1:
                    f1.writelines(Utterance_ID + ' ' + audio_path + '\n')