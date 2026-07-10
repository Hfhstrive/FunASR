workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path
model_name_or_model_dir="FunAudioLLM/Fun-ASR-Nano-2512"

# 病例描述：即术后一段话概述
train_data=/media/inno/ASR/ChatML/V4/train.jsonl
val_data=/media/inno/ASR/ChatML/V4/val.jsonl
output_dir="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"
echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool=`which funasr-train-ds`
echo "Using funasr trainer: ${train_tool}"

torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=8000  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=0 \
++train_conf.max_epoch=40 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=200 \
++train_conf.save_checkpoint_interval=200 \
++train_conf.early_stopping_patience=3 \
++train_conf.accum_grad=4 \
++train_conf.effective_save_name_excludes="None" \
++train_conf.keep_nbest_models=5 \
++train_conf.avg_nbest_model=3 \
++train_conf.use_deepspeed=false \
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=0.00003 \
++audio_encoder_conf.freeze=True \
++audio_adaptor_conf.freeze=false \
++llm_conf.freeze=false \
++output_dir="${output_dir}"
