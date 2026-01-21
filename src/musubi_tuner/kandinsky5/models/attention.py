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


def _maybe_compile(fn=None, **compile_kwargs):
    if not hasattr(_maybe_compile, "compile_targets"):
        _maybe_compile.compile_targets = []

    if fn is None:
        return lambda f: _maybe_compile(f, **compile_kwargs)

    # Create a wrapper so we can replace it later
    def wrapper(*args, **kwargs):
        return wrapper._fn(*args, **kwargs)

    wrapper._fn = fn
    wrapper._orig_fn = fn
    wrapper._compile_kwargs = compile_kwargs

    _maybe_compile.compile_targets.append(wrapper)

    return wrapper


def activate_compile(backend="inductor", mode="default", fullgraph=False, dynamic=None):
    if not hasattr(_maybe_compile, "compile_targets"):
        return

    for wrapper in _maybe_compile.compile_targets:
        if not hasattr(wrapper, "_compiled"):
            if wrapper._compile_kwargs == {}:  # empty dict so use our passed in ones
                wrapper._compile_kwargs = {"backend": backend, "fullgraph": fullgraph, "mode": mode, "dynamic": dynamic}
            wrapper._fn = torch.compile(wrapper._fn, **wrapper._compile_kwargs)
            wrapper._compiled = True


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
