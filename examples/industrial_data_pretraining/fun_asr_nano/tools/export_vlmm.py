import os
import glob
import json
import shutil
import torch
from safetensors.torch import save_file

# 微调模型根目录 (包含 model.pt 和 config.yaml)
model_dir = "/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_v4/"

# 导出 vLLM 格式的输出目录
output_dir = os.path.join(model_dir, "Qwen3-0.6B-vllm")
os.makedirs(output_dir, exist_ok=True)

# 1. 寻找基础模型的 Qwen3-0.6B 配置文件/Tokenizer 目录
# 优先从微调目录的 config.yaml 中自动解析 init_param_path
qwen_config_dir = None
config_yaml_path = os.path.join(model_dir, "config.yaml")

if os.path.exists(config_yaml_path):
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_yaml_path)
        llm_conf = OmegaConf.to_container(config.get("llm_conf", {}), resolve=True)
        init_path = llm_conf.get("init_param_path", "")
        if init_path and os.path.isdir(init_path):
            qwen_config_dir = init_path
    except Exception as e:
        print(f"解析 config.yaml 警告: {e}")

# 若未能在 config.yaml 中找到或解析失败，使用默认的基础模型缓存路径
if not qwen_config_dir or not os.path.exists(qwen_config_dir):
    default_base_path = "/home/inno/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/Qwen3-0.6B/"
    if os.path.exists(default_base_path):
        qwen_config_dir = default_base_path
    else:
        raise FileNotFoundError(f"未找到基础模型 Qwen3-0.6B 配置文件目录，请检查路径。")

print(f"--> 使用 LLM 配置文件目录: {qwen_config_dir}")

# 2. 复制 tokenizer 与 config 等依赖文件到 output_dir
for fname in os.listdir(qwen_config_dir):
    src = os.path.join(qwen_config_dir, fname)
    dst = os.path.join(output_dir, fname)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
print(f"--> 已复制配置文件和 Tokenizer 相关的 {len(os.listdir(output_dir))} 个文件")

# 3. 从微调导出的 model.pt 提取 'llm.' 前缀的微调后 LLM 权重
model_pt = os.path.join(model_dir, "model.pt")
if not os.path.exists(model_pt):
    raise FileNotFoundError(f"微调权重文件不存在: {model_pt}")

print(f"--> 正在读取微调权重 {model_pt} 并提取 LLM 权重...")
checkpoint = torch.load(model_pt, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)

llm_state = {}
for key, value in state_dict.items():
    if key.startswith("llm."):
        new_key = key[len("llm."):]  # 去掉 'llm.' 前缀
        # 使用 .clone() 避免 tie_word_embeddings (lm_head.weight 与 embed_tokens.weight) 共享内存报 safetensors 错误
        llm_state[new_key] = value.clone()

if not llm_state:
    raise RuntimeError("在 model.pt 中未找到带 'llm.' 前缀的 LLM 权重！")

print(f"--> 成功提取 {len(llm_state)} 个 LLM 权重 Tensor")

# 4. 保存为 vLLM 优先加载的 safetensors 格式
save_path = os.path.join(output_dir, "model.safetensors")
save_file(llm_state, save_path)

# 5. 写入 safetensors 索引文件
index = {
    "metadata": {"total_size": sum(v.numel() * v.element_size() for v in llm_state.values())},
    "weight_map": {k: "model.safetensors" for k in llm_state.keys()},
}
with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

print("=" * 60)
print(f" 转换成功！vLLM 模型格式已导出至: {output_dir}")
print("=" * 60)
