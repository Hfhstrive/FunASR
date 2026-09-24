#!/usr/bin/env bash
# Copyright FunASR.
# 针对唤醒词“鹰眼鹰眼”的轻量化 SANM-KWS 专用模型微调脚本

set -e
set -u

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${workspace}/../../..:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 1. 预训练底模文件路径（使用官方在线通用流式底模）
model_hub_dir="${HOME}/.cache/modelscope/hub/models/iic/speech_sanm_kws_phone-xiaoyun-commands-online"

config_name="sanm_6e_320_256_fdim40_t2602_yinyan.yaml"
token_list="${model_hub_dir}/tokens_2602.txt"
lexicon_list="${model_hub_dir}/lexicon.txt"
cmvn_file="${model_hub_dir}/am.mvn.dim40_l3r3"
init_param="${model_hub_dir}/basetrain_sanm_6e_320_256_fdim40_t2602_online.pt"

# 2. 已处理并平衡好的数据集路径 (必须使用纯英文软链接路径，规避 Hydra 对非 ASCII 中文字符的解析限制)
data_dir="/media/inno/ASR/KWS/TrainData/kws_v3"
train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

# 3. 输出模型目录
output_dir="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3"
mkdir -p "${output_dir}"

current_time=$(date "+%Y%m%d_%H%M%S")
log_file="${output_dir}/train_${current_time}.log"

echo "=================================================="
echo " 正在启动【鹰眼鹰眼】KWS 模型微调训练 "
echo "   - GPU              : ${CUDA_VISIBLE_DEVICES}"
echo "   - 配置文件         : ${config_name}"
echo "   - 训练底模         : ${init_param}"
echo "   - 训练数据         : ${train_data}"
echo "   - 验证数据         : ${val_data}"
echo "   - 输出目录         : ${output_dir}"
echo "   - 日志文件         : ${log_file}"
echo "=================================================="

# 4. 执行训练 (使用 torch2.7.1 虚拟环境的 torchrun 单卡启动)
export PATH="/home/inno/anaconda3/envs/torch2.7.1/bin:${PATH}"
export LD_LIBRARY_PATH="/home/inno/anaconda3/envs/torch2.7.1/lib:${LD_LIBRARY_PATH:-}"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
/home/inno/anaconda3/envs/torch2.7.1/bin/torchrun --nnodes 1 --nproc_per_node ${gpu_num} \
    "${workspace}/../../../funasr/bin/train.py" \
    --config-path "${workspace}/conf" \
    --config-name "${config_name}" \
    ++init_param="${init_param}" \
    ++disable_update=true \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++tokenizer_conf.token_list="${token_list}" \
    ++tokenizer_conf.seg_dict="${lexicon_list}" \
    ++frontend_conf.cmvn_file="${cmvn_file}" \
    ++output_dir="${output_dir}" \
    ++dataset_conf.batch_size=32 \
    ++train_conf.max_epoch=25 \
    ++train_conf.log_interval=10 \
    ++train_conf.validate_interval=2 \
    ++train_conf.save_checkpoint_interval=2 2>&1 | tee "${log_file}"

echo "=================================================="
echo " 微调训练结束！输出模型位于: ${output_dir}"
echo "=================================================="
