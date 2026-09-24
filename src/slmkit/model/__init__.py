"""The model: a Llama-style decoder. Specification in docs/MODEL.md."""

from slmkit.model.llama import IGNORE_INDEX, CausalLM, ModelArgs
from slmkit.model.stats import (
    CKPT_BYTES_PER_PARAM,
    PLANNING_TFLOPS,
    TRAIN_BYTES_PER_PARAM,
    ParamCount,
    count_parameters,
    estimated_gpu_hours,
    flops_per_token,
)

__all__ = [
    "CKPT_BYTES_PER_PARAM",
    "IGNORE_INDEX",
    "PLANNING_TFLOPS",
    "TRAIN_BYTES_PER_PARAM",
    "CausalLM",
    "ModelArgs",
    "ParamCount",
    "count_parameters",
    "estimated_gpu_hours",
    "flops_per_token",
]
