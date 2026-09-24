# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# 1. 指向微调产出目录中实际保存的完整 config.yaml (包含词表与 CMVN 路径)
config_path="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3"
config_file="config.yaml"

# 2. 权重路径与导出目录
model_path="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3/model.pt.best"
output_dir="/media/inno/work_dirs/ASR/KWS/kws_yingyan_v3/onnx/"

python -m funasr.bin.export \
    --config-path="${config_path}" \
    --config-name="${config_file}" \
    ++init_param=${model_path} \
    ++type="onnx" \
    ++output_dir="${output_dir}" \
    ++opset_version=14 \
    ++quantize=True
