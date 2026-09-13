"""Frozen llama.cpp keyword defaults used at the local runtime boundary."""

import types

MODEL_OPTIONS = types.MappingProxyType({
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "seed": 0,
    "verbose": False,
})
GENERATION_OPTIONS = types.MappingProxyType({
    "temperature": 0.0,
    "max_tokens": 1024,
    "seed": 0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
})
RESPONSE_FORMAT = "json_object"
