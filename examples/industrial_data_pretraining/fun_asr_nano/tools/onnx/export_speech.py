import os
import torch
import funasr.models.transformer.utils.nets_utils as nets_utils
import funasr.models.llm_asr.adaptor as adaptor
import funasr.models.transformer.attention as attention
import funasr.models.transformer.encoder as encoder
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

# Monkey Patch for make_pad_mask to support dynamic trace in ONNX
class ONNXMakePadMask(torch.nn.Module):
    def forward(self, lengths, xs=None, length_dim=-1, maxlen=None):
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths)
        if maxlen is not None:
            m = maxlen
        elif xs is not None:
            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            m = xs.shape[length_dim]
        else:
            m = lengths.max()
            
        if isinstance(m, torch.Tensor):
            m = m.to(torch.int64)
        elif isinstance(m, int):
            m = torch.tensor(m, dtype=torch.int64, device=lengths.device)
            
        # 预设一个足够大的常数 row_vector 避免在 arange 中使用动态 Tensor 导致 ONNX 追踪时折叠为常数
        row_vector = torch.arange(0, 3000, dtype=torch.int64, device=lengths.device)
        matrix = lengths.unsqueeze(-1)
        mask = row_vector >= matrix
        
        # 使用 m (Tensor) 进行动态切片，这在 ONNX 中会生成动态的 Slice 算子，支持动态序列长度
        mask = mask[:, :m]
        
        if xs is not None:
            ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
            mask = mask[ind].expand_as(xs).to(lengths.device)
        return mask

ONNX_MASK = ONNXMakePadMask()

nets_utils.make_pad_mask = ONNX_MASK
adaptor.make_pad_mask = ONNX_MASK
attention.make_pad_mask = ONNX_MASK
encoder.make_pad_mask = ONNX_MASK

print("Monkey Patch 状态验证:")
print("adaptor 中的 make_pad_mask ->", adaptor.make_pad_mask)
print("attention 中的 make_pad_mask ->", attention.make_pad_mask)

from funasr import AutoModel

finetune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/"
# 转为的model_speech.onnx为fp32，model_speech_int8.onnx为int8动态量化，model_speech_int4.onnx为int4权重仅量化
output_onnx_path = os.path.join(finetune_checkpoint, "model_speech.onnx")

print("正在载入微调模型...")
# 载入微调得到的模型，默认设备为 cuda:0
model = AutoModel(
    model=finetune_checkpoint,
    device="cuda:0",
    disable_update=True,
)

py_model = model.model

# 自定义用于导出的声学前端+适配器前向计算过程，输出 1024 维适配特征
def dynamic_forward_export(self, speech, speech_lengths):
    x, olens = self.audio_encoder(speech, speech_lengths)
    encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)
    return encoder_out, encoder_out_lens

import types
# 动态绑定 Method
py_model.forward = types.MethodType(dynamic_forward_export, py_model)
py_model.eval()

# 获取设备与模型精度类型
device = next(py_model.parameters()).device
dtype = next(py_model.parameters()).dtype

# 准备 Dummy 虚拟输入数据
speech = torch.randn(1, 100, 560).to(device=device, dtype=dtype)
speech_lengths = torch.tensor([100], dtype=torch.int32).to(device=device)

dummy_inputs = (speech, speech_lengths)

input_names = ["speech", "speech_lengths"]
output_names = ["encoder_out", "encoder_out_lens"]

# 配置动态轴参数
dynamic_axes = {
    "speech": {0: "batch", 1: "frames"},
    "speech_lengths": {0: "batch"},
    "encoder_out": {0: "batch", 1: "seq_len"},
    "encoder_out_lens": {0: "batch"}
}

print(f"正在导出声学前端模型为 ONNX 格式，保存路径: {output_onnx_path} ...")
with torch.no_grad():
    torch.onnx.export(
        py_model,
        dummy_inputs,
        output_onnx_path,
        verbose=True,
        do_constant_folding=True,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("ONNX 导出成功！")
    
    # 自动进行量化处理
    speech_int8_path = os.path.join(finetune_checkpoint, "model_speech_int8.onnx")
    speech_int4_path = os.path.join(finetune_checkpoint, "model_speech_int4.onnx")
    
    print("============================================================")
    print("【开始】动态 INT8 量化...")
    try:
        quantize_dynamic(
            model_input=output_onnx_path,
            model_output=speech_int8_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul']
        )
        print("【成功】INT8 动态量化完成，保存路径为:", speech_int8_path)
    except Exception as e:
        print("【失败】INT8 动态量化出错:", e)
        
    print("============================================================")
    print("【开始】权重仅 INT4 对称量化...")
    try:
        quantizer = MatMul4BitsQuantizer(
            model=output_onnx_path,
            block_size=128,
            is_symmetric=True
        )
        quantizer.process()
        # 使用 SerializeToString 直接二进制写盘，规避 onnx 官方库在 local function 量化保存时的 bug
        with open(speech_int4_path, "wb") as f:
            f.write(quantizer.model.model.SerializeToString())
        print("【成功】INT4 权重对称量化完成，保存路径为:", speech_int4_path)
    except Exception as e:
        print("【失败】INT4 权重对称量化出错:", e)
        
    print("============================================================")
    print("【声学模型量化大小对比结果】:")
    def print_size(name, path):
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"  - {name}: {size_mb:.2f} MB")
        else:
            print(f"  - {name}: 未生成")
            
    print_size("原始 FP32 模型", output_onnx_path)
    print_size("动态 INT8 模型", speech_int8_path)
    print_size("权重仅 INT4 模型", speech_int4_path)
    print("============================================================")

