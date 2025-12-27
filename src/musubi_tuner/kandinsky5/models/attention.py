# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as flash_attention_2
except:
    flash_attention_2 = None

try:
    from flash_attn_interface import flash_attn_func as flash_attention_3
except:
    flash_attention_3 = None

try:
    import sageattention
except:
    sageattention = None

try:
    import xformers.ops as xops
except:
    xops = None

_ENABLE_COMPILE = False


def set_compile_enabled(enabled: bool):
    global _ENABLE_COMPILE
    _ENABLE_COMPILE = bool(enabled)


def _maybe_compile(fn=None, **kwargs):
    if fn is None:
        return lambda f: _maybe_compile(f, **kwargs)
    if _ENABLE_COMPILE:
        return torch.compile(fn, **kwargs)
    return fn


@_maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def sdpa(q, k, v, attn_mask=None):
    query = q.transpose(1, 2).contiguous()
    key = k.transpose(1, 2).contiguous()
    value = v.transpose(1, 2).contiguous()
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask).transpose(1, 2).contiguous()
    return out


@_maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def sage_attn(q, k, v):
    out = sageattention.sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    return out


@_maybe_compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def xformers_attn(q, k, v, attn_mask=None):
    if attn_mask is not None:
        return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)
    return xops.memory_efficient_attention(q, k, v)


class SelfAttentionEngine:
    def __init__(self, engine="auto"):
        assert engine in ["auto", "flash_attention_2", "flash_attention_3", "sage", "sdpa", "xformers"]
        self.attention_fn = None
        self.supports_mask = False

        if engine == "flash_attention_2":
            if flash_attention_2 is None:
                raise RuntimeError("flash_attention_2 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_2
            self.supports_mask = False

        if engine == "flash_attention_3":
            if flash_attention_3 is None:
                raise RuntimeError("flash_attention_3 engine selected, but it can't be imported.")
            self.attention_fn = flash_attention_3
            self.supports_mask = False

        if engine == "sage":
            if sageattention is None:
                raise RuntimeError("sage engine selected, but it can't be imported.")
            self.attention_fn = sage_attn
            self.supports_mask = False

        if engine == "xformers":
            if xops is None:
                raise RuntimeError("xformers engine selected, but it can't be imported.")
            self.attention_fn = xformers_attn
            self.supports_mask = False

        if engine == "sdpa":
            self.attention_fn = sdpa
            self.supports_mask = True

        if engine == "auto":
            self.attention_fn = sdpa
            self.supports_mask = True
            if xops is not None:
                self.attention_fn = xformers_attn
                self.supports_mask = False
            if sageattention is not None:
                self.attention_fn = sage_attn
                self.supports_mask = False
            if flash_attention_2 is not None:
                self.attention_fn = flash_attention_2
                self.supports_mask = False
            if flash_attention_3 is not None:
                self.attention_fn = flash_attention_3
                self.supports_mask = False

    def get_attention(self):
        return self.attention_fn
