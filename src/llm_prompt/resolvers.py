import torch
from omegaconf import OmegaConf

DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

OmegaConf.register_new_resolver("dtype", lambda x: DTYPE_MAPPING[x])
