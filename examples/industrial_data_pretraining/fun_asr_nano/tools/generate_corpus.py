import glob
import os
import random
from ipdb import set_trace


def corpus_convert(word_path, speech_path, save_path):
    # 遍历word_path 或 speech_path 下的文件夹，根据文件夹划分数据集(训练、测试)
    for lesion in os.listdir(word_path):
        train_nums, val_nums = 0, 0
        work_lesion_path = os.path.join(word_path, lesion)
        lesion_words = glob.glob(f'{work_lesion_path}/**.txt')
        for lesion_word in lesion_words:
            # 将case_no 作为 Utterance_ID：
            case_no = lesion_word.split('/')[-1].split('.')[0]
            with open(lesion_word, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1
            word_info = lines[0].strip('\n')
            lesion_speech = lesion_word.replace(word_path, speech_path).replace('.txt', '.mp3')
            assert os.path.exists(lesion_speech)
            if val_nums >= len(lesion_words) * 0.1:
                mode = 'train'
                train_nums += 1
            elif val_nums == 0 or random.random() <= 0.1:
                mode = 'val'
                val_nums += 1
            else:
                mode = 'train'
                train_nums += 1
            save_scp_path = os.path.join(save_path, mode + '.scp')
            save_txt_path = os.path.join(save_path, mode + '.txt')
            with open(save_scp_path, 'a+') as f1:
                f1.writelines(case_no + ' ' + lesion_speech + '\n')
            with open(save_txt_path, 'a+') as f1:
                f1.writelines(case_no + ' ' + word_info + '\n')


if __name__ == '__main__':
    random.seed(20260528)
    word_path = '/media/inno/ASR/base_data/oral/case/'
    speech_path = '/media/inno/ASR/audio/train/real/case/'
    save_path = '/media/inno/ASR/ChatML/V3/'
    os.makedirs(save_path, exist_ok=True)
    corpus_convert(word_path, speech_path, save_path)