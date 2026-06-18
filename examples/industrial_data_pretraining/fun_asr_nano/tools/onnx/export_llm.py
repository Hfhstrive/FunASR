import os
import torch
import torch.nn as nn
import numpy as np
from funasr import AutoModel
import transformers.masking_utils as masking_utils
from typing import Optional

# 1. Monkey patch for create_causal_mask to bypass functorch vmap in ONNX tracing
def onnx_create_causal_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids=None,
    or_mask_function=None,
    and_mask_function=None,
):
    batch_size = input_embeds.shape[0]
    seq_len = input_embeds.shape[1]
    dtype = input_embeds.dtype
    device = input_embeds.device

    # Standard causal mask: shape [seq_len, seq_len]
    row_vector = torch.arange(0, 3000, dtype=torch.int64, device=device)
    row_vector = row_vector[:seq_len]
    col_vector = row_vector.unsqueeze(1)
    causal_matrix = col_vector >= row_vector # True where col >= row
    
    # padding mask: shape [batch_size, seq_len]
    if attention_mask is not None:
        padding_matrix = attention_mask.to(torch.bool).unsqueeze(1) # shape [batch_size, 1, seq_len]
        combined_mask = causal_matrix.unsqueeze(0) & padding_matrix.unsqueeze(2) # shape [batch_size, 1, seq_len, seq_len]
    else:
        combined_mask = causal_matrix.unsqueeze(0).unsqueeze(1)
        
    combined_mask = combined_mask.to(torch.bool)
    min_val = -10000.0  # Safe float representation of -inf for ONNX
    float_mask = torch.where(combined_mask, torch.tensor(0.0, dtype=torch.float32, device=device), torch.tensor(min_val, dtype=torch.float32, device=device))
    return float_mask.to(dtype)

masking_utils.create_causal_mask = onnx_create_causal_mask

# 2. Monkey patch for eager_attention_forward for numerical stability (Attention Upcasting) in FP16/BF16
import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling

def onnx_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = qwen3_modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen3_modeling.repeat_kv(value, module.num_key_value_groups)

    # 提升注意力计算的中间步骤（矩阵乘法与 scale）到 float32 高精度以防溢出
    query_fp32 = query.to(torch.float32)
    key_states_fp32 = key_states.to(torch.float32)
    value_states_fp32 = value_states.to(torch.float32)

    attn_weights = torch.matmul(query_fp32, key_states_fp32.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask.to(torch.float32)

    # 保持 softmax 在 float32 计算
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states_fp32)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # 最终转回输入 Tensor 的原始精度类型 (如 float16 或 bfloat16)，因为 query.dtype 可能会在 RoPE 作用后被 PyTorch 提升为 float32
    return attn_output.to(value.dtype), attn_weights.to(value.dtype)

qwen3_modeling.eager_attention_forward = onnx_eager_attention_forward


class Qwen3Wrapper(nn.Module):
    def __init__(self, llm, export_dtype, output_dtype=torch.float32):
        super().__init__()
        self.llm = llm
        self.export_dtype = export_dtype
        self.output_dtype = output_dtype
        self.num_layers = llm.config.num_hidden_layers
        
    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
        # 仅在数据类型不一致时进行 Cast，以防 PyTorch Trace 机制强行将外部输入接口提升为 float32
        if inputs_embeds.dtype != self.export_dtype:
            inputs_embeds_cast = inputs_embeds.to(self.export_dtype)
        else:
            inputs_embeds_cast = inputs_embeds
        
        # 重构打平的 past_key_values 为元组
        reconstructed_past = ()
        for i in range(self.num_layers):
            k = past_key_values[2 * i]
            v = past_key_values[2 * i + 1]
            if k.dtype != self.export_dtype:
                k = k.to(self.export_dtype)
                v = v.to(self.export_dtype)
            reconstructed_past += ((k, v),)
            
        import transformers
        # 包装成新版 transformers 期待的 DynamicCache 缓存容器，规避属性缺失报错
        past_key_values_cache = transformers.cache_utils.DynamicCache.from_legacy_cache(reconstructed_past)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds_cast,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values_cache,
            use_cache=True,
            return_dict=False
        )
        logits = outputs[0]
        present_key_values = outputs[1]
        
        # 还原回传统的 legacy 元组格式以打平输出
        present_legacy = present_key_values.to_legacy_cache()
        flat_outputs = [logits.to(self.output_dtype)]
        for i in range(self.num_layers):
            present_k = present_legacy[i][0].to(self.output_dtype)
            present_v = present_legacy[i][1].to(self.output_dtype)
            flat_outputs.append(present_k)
            flat_outputs.append(present_v)
            
        return tuple(flat_outputs)

def main(args):
    finetune_checkpoint = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/"
    print("正在加载模型到 cuda:0 ...")
    model = AutoModel(
        model=finetune_checkpoint,
        device="cuda:0",
        disable_update=True,
    )
    
    py_model = model.model
    llm = py_model.llm
    
    if args.precision == "BF16":
        print("【模式】将模型参数与配置转为 Bfloat16...")
        llm = llm.to(torch.bfloat16)
        export_dtype = torch.bfloat16
        output_dtype = torch.float32  # 维持 float32 隔离层以兼容 numpy
    elif args.precision == "FP16":
        print("【模式】将模型参数与配置转为 Float16...")
        llm = llm.to(torch.float16)
        export_dtype = torch.float16
        output_dtype = torch.float16  # 原生 FP16 接口以优化显存
        
        # 对 LM Head 的前向计算进行高精度提升 (Upcasting)，防止 15 万维分类累加在 FP16 下数值溢出
        import types
        def patched_lm_head_forward(self, x):
            bias_cast = self.bias.to(torch.float32) if self.bias is not None else None
            return nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32), bias_cast)
        llm.lm_head.forward = types.MethodType(patched_lm_head_forward, llm.lm_head)
        print("【混合精度】已成功对 lm_head 应用高精度前向计算补丁...")

        # 对 MLP 的前向计算进行高精度提升 (Upcasting)，防止 SwiGLU 中间乘积及 down_proj 累加溢出，并保持输出为 float32
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        def patched_mlp_forward(self, x):
            x_fp32 = x.to(torch.float32)
            gate_out = self.act_fn(nn.functional.linear(x_fp32, self.gate_proj.weight.to(torch.float32), None))
            up_out = nn.functional.linear(x_fp32, self.up_proj.weight.to(torch.float32), None)
            prod = gate_out * up_out
            down_out = nn.functional.linear(prod, self.down_proj.weight.to(torch.float32), None)
            return down_out
        qwen3_modeling.Qwen3MLP.forward = patched_mlp_forward

        # 对 Decoder Layer 的前向传播进行高精度残差流提升，防止层间累加溢出
        def patched_decoder_layer_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor:
            hidden_states_fp32 = hidden_states.to(torch.float32)
            residual = hidden_states_fp32
            attn_in = self.input_layernorm(hidden_states_fp32).to(torch.float16)
            attn_outputs = self.self_attn(
                hidden_states=attn_in,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            attn_output = attn_outputs[0]
            hidden_states_fp32 = residual + attn_output.to(torch.float32)

            residual = hidden_states_fp32
            mlp_in = self.post_attention_layernorm(hidden_states_fp32).to(torch.float16)
            mlp_output = self.mlp(mlp_in)
            hidden_states_fp32 = residual + mlp_output.to(torch.float32)
            return hidden_states_fp32

        qwen3_modeling.Qwen3DecoderLayer.forward = patched_decoder_layer_forward
        print("【混合精度】已成功应用残差流高精度提升补丁 (Upcasted Residual Stream)...")
    else:
        print("【模式】将模型参数与配置转为 Float32...")
        llm = llm.to(torch.float32)
        export_dtype = torch.float32
        output_dtype = torch.float32

    llm.config._attn_implementation = "eager"
    llm.config._attn_implementation_internal = "eager"
    llm.eval()
    
    wrapper = Qwen3Wrapper(llm, export_dtype, output_dtype)
    wrapper.eval()
    
    device = next(llm.parameters()).device
    dtype = next(llm.parameters()).dtype
    print(f"模型设备: {device}, 精度类型: {dtype}")
    
    # 保存词嵌入 Word Embedding 权重到 .npy 文件，采用与导出一致的精度
    print("提取并保存词向量权重 (embed_tokens.npy) ...")
    if args.precision == "BF16":
        import ml_dtypes
        embed_weight = llm.model.embed_tokens.weight.detach().cpu().to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
    elif args.precision == "FP16":
        embed_weight = llm.model.embed_tokens.weight.detach().cpu().to(torch.float16).numpy()
    else:
        embed_weight = llm.model.embed_tokens.weight.detach().cpu().to(export_dtype).numpy()
    np.save(os.path.join(finetune_checkpoint, "embed_tokens.npy"), embed_weight)
    print(f"词向量提取完成, 维度为 {embed_weight.shape}, 精度为 {embed_weight.dtype}, 保存成功。")
    
    # 3. 准备 Dummy 虚拟输入数据
    # batch=1, seq_len=10, hidden_size=1024
    if args.precision == "FP16":
        dummy_dtype = torch.float16
    else:
        dummy_dtype = torch.float32
        
    inputs_embeds = torch.randn(1, 10, 1024, device=device, dtype=dummy_dtype)
    attention_mask = torch.ones(1, 10, device=device, dtype=torch.int32)
    position_ids = torch.arange(0, 10, dtype=torch.int64, device=device).unsqueeze(0)
    
    # 构造 56 个 dummy past_key_values 分量
    # 形状为 [batch=1, num_key_value_heads=8, past_seq_len=0, head_dim=128]，类型与 dummy_dtype 一致
    past_key_values_dummy = []
    num_layers = llm.config.num_hidden_layers
    for i in range(num_layers):
        dummy_k = torch.randn(1, 8, 0, 128, device=device, dtype=dummy_dtype)
        dummy_v = torch.randn(1, 8, 0, 128, device=device, dtype=dummy_dtype)
        past_key_values_dummy.extend([dummy_k, dummy_v])
        
    print("正在导出 LLM (Qwen3) 到 ONNX 格式...")
    output_onnx_path = os.path.join(finetune_checkpoint, "model_llm.onnx")
    
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")
        
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")
        
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"}
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "present_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "present_seq_len"}
        
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (inputs_embeds, attention_mask, position_ids, *past_key_values_dummy),
            output_onnx_path,
            verbose=False,
            do_constant_folding=True,
            opset_version=14,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        
    print("PyTorch ONNX 导出完成，正在检测导出模型大小并进行优化打包保存...")
    
    import onnx
    # 动态获取磁盘文件字节数（包含可能被自动拆分的外部权重分卷），防止超过 2GB 时 Protobuf 内存序列化崩溃
    total_size = os.path.getsize(output_onnx_path)
    for f in os.listdir(finetune_checkpoint):
        if f.startswith("onnx__") or f.startswith("Constant_"):
            total_size += os.path.getsize(os.path.join(finetune_checkpoint, f))
    total_size_gb = total_size / (1024 ** 3)
    print(f"最终打包模型总大小: {total_size_gb:.3f} GB (字节数: {total_size})")
    
    clean_temp_files = True
    limit_2gb = 2.0 * 1024 * 1024 * 1024  # 2 GB
    if total_size < limit_2gb:
        print("模型大小小于 2GB，打包为单一 .onnx 文件 (不需要 model_llm.onnx.data) ...")
        # 已经由 torch.onnx.export 写入为单一文件，我们只需清理旧的分卷数据包残留
        old_data_file = os.path.join(finetune_checkpoint, "model_llm.onnx.data")
        if os.path.exists(old_data_file):
            try:
                os.remove(old_data_file)
            except Exception:
                pass
    else:
        print("模型大小超过 2GB，打包权重文件为单外部数据包 (model_llm.onnx.data) ...")
        try:
            model_proto = onnx.load(output_onnx_path)
            onnx.save_model(
                model_proto,
                output_onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model_llm.onnx.data"
            )
            print("模型外部权重分卷处理成功！")
        except Exception as e:
            print(f"警告: ONNX 外部权重转换失败 ({e})。已保留原始导出的外部权重分卷，模型依旧可用。")
            clean_temp_files = False
        
    if clean_temp_files:
        print("打包与保存完毕。正在清理临时分散权重文件...")
        for f in os.listdir(finetune_checkpoint):
            if f.startswith("onnx__") or f.startswith("Constant_"):
                try:
                    os.remove(os.path.join(finetune_checkpoint, f))
                except Exception:
                    pass
    else:
        print("已保留所有权重分卷。ONNX 路径:", output_onnx_path)
                
    print("LLM 端 ONNX 处理成功！")
    print("ONNX 路径:", output_onnx_path)
    if total_size >= limit_2gb:
        print("权重外部路径:", os.path.join(finetune_checkpoint, "model_llm.onnx.data"))
    else:
        print("所有权重已直接包含在 .onnx 文件中。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM ONNX 导出脚本")
    parser.add_argument(
        "--precision",
        type=str,
        default="FP32",
        choices=["FP16", "BF16", "FP32", "fp16", "bf16", "fp32"],
        help="导出精度模式 (FP16, BF16, FP32)"
    )
    args = parser.parse_args()
    args.precision = args.precision.upper()
    main(args)
