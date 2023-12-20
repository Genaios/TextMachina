from enum import Enum
from typing import Final

import torch
from transformers import BitsAndBytesConfig

GENERATION_ERROR: Final = "<error>"

QUANTIZATION_CONFIGS: Final = {
    "int4": {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)},
    "int4_bf16": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    },
    "int4_nf4": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4"
        )
    },
    "int8": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
    "int4_nested": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True
        )
    },
    "fp16": {"torch_dtype": torch.float16},
    "none": {},
}


class CompletionType(str, Enum):
    CHAT: str = "CHAT"
    COMPLETION: str = "COMPLETION"
