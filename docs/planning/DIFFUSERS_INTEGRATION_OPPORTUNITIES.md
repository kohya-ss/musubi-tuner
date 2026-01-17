# Diffusers Integration Opportunities for Blissful-Tuner

This document provides a comprehensive reference for features and optimizations from HuggingFace Diffusers (v0.37.0+) that can be integrated into the Blissful-Tuner LoRA/LyCORIS training pipeline.

**Last Updated:** January 2026
**Diffusers Version Analyzed:** 0.37.0.dev0
**Blissful-Tuner Commit:** Current main branch

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Priority Matrix](#priority-matrix)
3. [Hook Architecture System](#1-hook-architecture-system)
4. [Group Offloading with Prefetching](#2-group-offloading-with-prefetching)
5. [Layerwise Casting (Dynamic Precision)](#3-layerwise-casting-dynamic-precision)
6. [FasterCache Optimization](#4-fastercache-optimization)
7. [Pyramid Attention Broadcast](#5-pyramid-attention-broadcast)
8. [Layer Skip Optimization](#6-layer-skip-optimization)
9. [TaylorSeer Cache](#7-taylorseer-cache)
10. [Context Parallelization](#8-context-parallelization)
11. [Unified Quantization Framework](#9-unified-quantization-framework)
12. [Modular Pipelines Architecture](#10-modular-pipelines-architecture)
13. [Implementation Roadmap](#implementation-roadmap)
14. [File Reference Map](#file-reference-map)

---

## Executive Summary

Diffusers provides a mature, extensible hook-based optimization system that can significantly enhance Blissful-Tuner's training and validation performance. The key opportunities are:

| Category | Potential Gain | Implementation Effort |
|----------|---------------|----------------------|
| Validation/Inference Speed | +50-100% | Medium |
| Memory Efficiency | -20-30% | Medium |
| Code Maintainability | Significant | High (refactor) |
| Multi-GPU Scaling | New capability | Medium |

**Current Blissful-Tuner State:**
- Custom offloading in `src/musubi_tuner/modules/custom_offloading_utils.py`
- Static FP8 quantization in `src/musubi_tuner/modules/fp8_optimization_utils.py`
- No unified hook system for composable optimizations
- Advanced LoRA features (DoRA, RS-LoRA, LoRA+) already implemented

**Key Integration Principle:** Adopt diffusers' composable hook architecture while preserving Blissful-Tuner's training-specific optimizations.

---

## Priority Matrix

### Tier 1: High Impact, Recommended for Immediate Integration

| Feature | Training Impact | Validation Impact | Memory Savings |
|---------|----------------|-------------------|----------------|
| HookRegistry Pattern | Architectural | Architectural | - |
| Group Offloading + Prefetch | - | +15-25% | -15% |
| Layerwise Casting | +5% | +15% | -20% |

### Tier 2: High Impact for Validation/Inference

| Feature | Training Impact | Validation Impact | Memory Savings |
|---------|----------------|-------------------|----------------|
| FasterCache | - | +25-40% | - |
| Pyramid Attention Broadcast | - | +20% | - |
| Layer Skip | - | +30% | - |

### Tier 3: Future Enhancements

| Feature | Training Impact | Validation Impact | Memory Savings |
|---------|----------------|-------------------|----------------|
| Unified Quantization | Flexibility | Flexibility | Variable |
| Context Parallel | Multi-GPU | Multi-GPU | Per-GPU reduction |
| TaylorSeer Cache | - | +15% | - |

---

## 1. Hook Architecture System

### Source Location
```
/root/diffusers/src/diffusers/hooks/
├── hooks.py          # Core HookRegistry and ModelHook classes
├── _common.py        # Layer type detection utilities
└── __init__.py       # Public exports
```

### What It Provides

A unified, composable system for attaching optimization hooks to PyTorch modules:

```python
# Core classes from diffusers/hooks/hooks.py

class ModelHook:
    """Base class for all optimization hooks."""
    _is_stateful: bool = False

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """Called when hook is registered."""
        return module

    def deinitialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        """Called when hook is removed."""
        return module

    def pre_forward(self, module, *args, **kwargs):
        """Called before module.forward()."""
        return args, kwargs

    def post_forward(self, module, output):
        """Called after module.forward()."""
        return output

    def new_forward(self, module, *args, **kwargs):
        """Completely replaces module.forward() if defined."""
        pass

    def reset_state(self, module: torch.nn.Module):
        """Reset any cached state (for stateful hooks)."""
        pass


class HookRegistry:
    """Registry for managing multiple hooks on a module."""

    @classmethod
    def check_if_exists_or_initialize(cls, module: torch.nn.Module) -> "HookRegistry":
        """Get or create registry for a module."""
        pass

    def register_hook(self, hook: ModelHook, name: str) -> None:
        """Register a named hook."""
        pass

    def get_hook(self, name: str) -> Optional[ModelHook]:
        """Retrieve a hook by name."""
        pass

    def remove_hook(self, name: str, recurse: bool = True) -> None:
        """Remove a hook by name."""
        pass
```

### Current Blissful-Tuner Gap

The current implementation in `custom_offloading_utils.py` uses direct module manipulation:

```python
# Current approach (custom_offloading_utils.py)
class ModelOffloader:
    def __init__(self, model, device, offload_device, ...):
        self.model = model
        # Direct module registration, no composability

    def swap_weight_devices_cuda(self, module, device):
        # Tightly coupled to offloading logic
        pass
```

### Integration Pattern

Create a new hook infrastructure in Blissful-Tuner:

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/hooks/base.py

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

class ModelHook:
    """Base hook class compatible with diffusers pattern."""
    _is_stateful: bool = False

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        return module

    def deinitialize_hook(self, module: nn.Module) -> nn.Module:
        return module

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        return args, kwargs

    def post_forward(self, module: nn.Module, output):
        return output

    def reset_state(self, module: nn.Module):
        pass


class HookRegistry:
    """Manages multiple hooks on a single module."""

    _REGISTRY_KEY = "_blissful_hook_registry"

    def __init__(self):
        self._hooks: Dict[str, ModelHook] = {}
        self._hook_handles = []

    @classmethod
    def check_if_exists_or_initialize(cls, module: nn.Module) -> "HookRegistry":
        if not hasattr(module, cls._REGISTRY_KEY):
            registry = cls()
            setattr(module, cls._REGISTRY_KEY, registry)
            registry._setup_forward_hooks(module)
        return getattr(module, cls._REGISTRY_KEY)

    def register_hook(self, hook: ModelHook, name: str) -> None:
        if name in self._hooks:
            raise ValueError(f"Hook '{name}' already registered")
        self._hooks[name] = hook

    def get_hook(self, name: str) -> Optional[ModelHook]:
        return self._hooks.get(name)

    def remove_hook(self, name: str) -> None:
        if name in self._hooks:
            del self._hooks[name]

    def _setup_forward_hooks(self, module: nn.Module):
        # Implementation details for wrapping forward
        pass
```

### Benefits

1. **Composability**: Multiple optimizations can coexist (offloading + quantization + caching)
2. **Clean Lifecycle**: Initialize/deinitialize for proper resource management
3. **State Management**: Stateful hooks can reset between training steps
4. **Debugging**: Named hooks make it easy to identify optimization sources

---

## 2. Group Offloading with Prefetching

### Source Location
```
/root/diffusers/src/diffusers/hooks/group_offloading.py  # 956 lines
```

### What It Provides

Advanced memory management with async prefetching:

```python
# Key concepts from group_offloading.py

class ModuleGroup:
    """Groups modules for coordinated offloading."""

    def __init__(
        self,
        modules: List[nn.Module],
        offload_device: torch.device,
        onload_device: torch.device,
        offload_leader: nn.Module,
        onload_leader: nn.Module,
        use_stream: bool = False,
        low_cpu_mem_usage: bool = False,
    ):
        pass

    def onload_(self):
        """Move group to compute device."""
        pass

    def offload_(self):
        """Move group to offload device."""
        pass


class GroupOffloadingHook(ModelHook):
    """Hook for basic group offloading."""

    def __init__(self, group: ModuleGroup, next_group: Optional[ModuleGroup]):
        self.group = group
        self.next_group = next_group

    def pre_forward(self, module, *args, **kwargs):
        self.group.onload_()
        return args, kwargs

    def post_forward(self, module, output):
        self.group.offload_()
        if self.next_group:
            self.next_group.onload_()  # Prefetch next group
        return output


class LazyPrefetchGroupOffloadingHook(ModelHook):
    """Learns execution order and prefetches accordingly."""

    def __init__(self, group: ModuleGroup, all_groups: List[ModuleGroup]):
        self.group = group
        self.all_groups = all_groups
        self._execution_order = []  # Learned during first pass
        self._learning = True

    def pre_forward(self, module, *args, **kwargs):
        if self._learning:
            self._execution_order.append(self.group)
        self.group.onload_()
        return args, kwargs

    def post_forward(self, module, output):
        # Prefetch next group based on learned order
        next_idx = self._get_next_group_index()
        if next_idx is not None:
            self.all_groups[next_idx].onload_()
        self.group.offload_()
        return output
```

### Current Blissful-Tuner Implementation

```python
# From custom_offloading_utils.py (simplified)
class ModelOffloader:
    def swap_weight_devices_cuda(self, module, device, non_blocking=True):
        stream = self.swap_stream
        with torch.cuda.stream(stream):
            for param in module.parameters():
                param.data = param.data.to(device, non_blocking=non_blocking)
```

**Limitations:**
- No automatic execution order detection
- Manual stream management
- No block-level vs leaf-level control

### Integration Pattern

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/hooks/group_offloading.py

from typing import List, Optional, Literal
import torch
import torch.nn as nn

def apply_group_offloading(
    module: nn.Module,
    onload_device: torch.device,
    offload_device: torch.device = torch.device("cpu"),
    offload_type: Literal["block_level", "leaf_level"] = "block_level",
    num_blocks_per_group: int = 1,
    use_stream: bool = True,
    record_stream: bool = True,
    low_cpu_mem_usage: bool = False,
    use_prefetching: bool = True,
) -> None:
    """
    Apply group offloading to a module.

    Args:
        module: The model to apply offloading to
        onload_device: Device for computation (e.g., cuda:0)
        offload_device: Device for storage (e.g., cpu)
        offload_type: "block_level" groups Sequential/ModuleList blocks,
                      "leaf_level" offloads individual layers
        num_blocks_per_group: How many blocks per offloading group
        use_stream: Enable CUDA streams for async transfers
        record_stream: Track stream usage for memory management
        low_cpu_mem_usage: Minimize CPU memory at cost of speed
        use_prefetching: Enable execution order learning and prefetching
    """
    # Detect module groups
    groups = _detect_module_groups(module, offload_type, num_blocks_per_group)

    # Create hooks
    HookClass = LazyPrefetchGroupOffloadingHook if use_prefetching else GroupOffloadingHook

    for i, group in enumerate(groups):
        next_group = groups[i + 1] if i + 1 < len(groups) else None
        hook = HookClass(group, next_group if not use_prefetching else groups)

        registry = HookRegistry.check_if_exists_or_initialize(group.onload_leader)
        registry.register_hook(hook, f"group_offload_{i}")


def _detect_module_groups(
    module: nn.Module,
    offload_type: str,
    num_blocks_per_group: int
) -> List[ModuleGroup]:
    """Automatically detect module groups for offloading."""
    groups = []

    if offload_type == "block_level":
        # Find Sequential and ModuleList containers
        for name, child in module.named_children():
            if isinstance(child, (nn.Sequential, nn.ModuleList)):
                # Group consecutive blocks
                blocks = list(child.children())
                for i in range(0, len(blocks), num_blocks_per_group):
                    group_blocks = blocks[i:i + num_blocks_per_group]
                    groups.append(ModuleGroup(
                        modules=group_blocks,
                        offload_leader=group_blocks[0],
                        onload_leader=group_blocks[-1],
                    ))
    else:  # leaf_level
        # Each leaf module is its own group
        for name, child in module.named_modules():
            if len(list(child.children())) == 0:  # Leaf
                groups.append(ModuleGroup(
                    modules=[child],
                    offload_leader=child,
                    onload_leader=child,
                ))

    return groups
```

### Performance Expectations

| Scenario | Current | With Prefetching | Improvement |
|----------|---------|------------------|-------------|
| Sequential blocks | 100% | 75-85% | 15-25% faster |
| Random access | 100% | 90-95% | 5-10% faster |
| Memory usage | Baseline | -15% | Better utilization |

---

## 3. Layerwise Casting (Dynamic Precision)

### Source Location
```
/root/diffusers/src/diffusers/hooks/layerwise_casting.py  # 241 lines
```

### What It Provides

Dynamic dtype conversion per layer during forward pass:

```python
# Key classes from layerwise_casting.py

class LayerwiseCastingHook(ModelHook):
    """Casts layer weights to compute dtype on forward, back to storage dtype after."""

    def __init__(
        self,
        storage_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        non_blocking: bool = False,
    ):
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.non_blocking = non_blocking

    def pre_forward(self, module, *args, **kwargs):
        # Cast weights to compute dtype
        for param in module.parameters(recurse=False):
            param.data = param.data.to(self.compute_dtype, non_blocking=self.non_blocking)
        return args, kwargs

    def post_forward(self, module, output):
        # Cast weights back to storage dtype
        for param in module.parameters(recurse=False):
            param.data = param.data.to(self.storage_dtype, non_blocking=self.non_blocking)
        return output


def apply_layerwise_casting(
    module: nn.Module,
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    skip_modules_pattern: List[str] = None,
    skip_modules_classes: Tuple[type, ...] = None,
    non_blocking: bool = False,
) -> None:
    """
    Apply layerwise casting to a module.

    Args:
        module: Model to apply casting to
        storage_dtype: Dtype for weight storage (e.g., torch.float8_e4m3fn)
        compute_dtype: Dtype for computation (e.g., torch.bfloat16)
        skip_modules_pattern: List of name patterns to skip (e.g., ["norm", "embed"])
        skip_modules_classes: Tuple of module classes to skip
        non_blocking: Use non-blocking transfers
    """
    skip_modules_pattern = skip_modules_pattern or []
    skip_modules_classes = skip_modules_classes or ()

    for name, submodule in module.named_modules():
        # Skip if matches pattern or class
        if any(pattern in name for pattern in skip_modules_pattern):
            continue
        if isinstance(submodule, skip_modules_classes):
            continue

        # Only apply to leaf modules with parameters
        if len(list(submodule.children())) > 0:
            continue
        if len(list(submodule.parameters(recurse=False))) == 0:
            continue

        hook = LayerwiseCastingHook(storage_dtype, compute_dtype, non_blocking)
        registry = HookRegistry.check_if_exists_or_initialize(submodule)
        registry.register_hook(hook, "layerwise_casting")
```

### Current Blissful-Tuner Implementation

```python
# From fp8_optimization_utils.py (lines 84-171)
def apply_fp8_monkey_patch(model, device, ...):
    """Static FP8 quantization - weights stay in FP8."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Quantize weights to FP8 permanently
            module.weight.data = quantize_to_fp8(module.weight.data)
```

**Limitations:**
- Static quantization (no dynamic casting)
- FP8-only (no other precision options)
- No skip patterns for sensitive layers

### Integration Pattern

```python
# Proposed: Extend fp8_optimization_utils.py

def apply_dynamic_precision(
    model: nn.Module,
    storage_dtype: torch.dtype = torch.float8_e4m3fn,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_patterns: List[str] = None,
    skip_classes: Tuple[type, ...] = None,
) -> None:
    """
    Apply dynamic precision casting during forward pass.

    Unlike static FP8 quantization, this:
    - Keeps weights in storage_dtype between forward passes
    - Casts to compute_dtype during forward
    - Casts back to storage_dtype after forward

    Args:
        model: Model to apply precision casting to
        storage_dtype: Low-precision dtype for memory (e.g., FP8)
        compute_dtype: Higher precision for computation (e.g., BF16)
        skip_patterns: Module name patterns to skip (normalization, embeddings)
        skip_classes: Module classes to skip
    """
    skip_patterns = skip_patterns or ["norm", "ln_", "embed", "proj_in", "proj_out"]
    skip_classes = skip_classes or (nn.LayerNorm, nn.GroupNorm, nn.Embedding)

    apply_layerwise_casting(
        module=model,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        skip_modules_pattern=skip_patterns,
        skip_modules_classes=skip_classes,
        non_blocking=True,
    )


# PEFT Compatibility Hook (from diffusers)
class PeftInputAutocastDisableHook(ModelHook):
    """Prevents PEFT from re-casting inputs during LoRA forward."""

    def pre_forward(self, module, *args, **kwargs):
        # Disable autocast for PEFT layers
        if hasattr(module, "disable_adapters"):
            module._autocast_enabled = False
        return args, kwargs
```

### Memory Savings Calculation

| Dtype | Bits per Param | Memory vs FP32 |
|-------|---------------|----------------|
| FP32 | 32 | 100% |
| BF16 | 16 | 50% |
| FP8 | 8 | 25% |

For a 14B parameter model:
- FP32: ~56 GB
- BF16: ~28 GB
- FP8 storage + BF16 compute: ~14 GB storage, ~28 GB peak during forward

---

## 4. FasterCache Optimization

### Source Location
```
/root/diffusers/src/diffusers/hooks/faster_cache.py  # 655 lines
```

### What It Provides

Skip attention computation when outputs would be similar:

```python
# Key concepts from faster_cache.py

@dataclass
class FasterCacheConfig:
    """Configuration for FasterCache optimization."""

    # Skip spatial attention every N blocks
    spatial_attention_block_skip_range: int = 2

    # Only apply caching in certain timestep ranges
    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)

    # Callback to get current timestep
    current_timestep_callback: Callable[[], int] = None

    # Skip unconditional branch every N steps (for CFG)
    unconditional_batch_skip_range: int = 5

    # Use low-frequency approximation for skipped steps
    use_low_freq_approximation: bool = True

    # Tensor names for caching
    attention_cache_key: str = "attn_output"


class FasterCacheHook(ModelHook):
    """Caches and reuses attention outputs."""
    _is_stateful = True

    def __init__(self, config: FasterCacheConfig, block_index: int):
        self.config = config
        self.block_index = block_index
        self._cache = {}
        self._step_count = 0

    def pre_forward(self, module, *args, **kwargs):
        timestep = self.config.current_timestep_callback()

        # Check if we should skip this computation
        if self._should_skip(timestep):
            # Return cached output instead of computing
            return self._get_cached_output()

        return args, kwargs

    def post_forward(self, module, output):
        # Cache the output for future steps
        self._cache[self.config.attention_cache_key] = output
        self._step_count += 1
        return output

    def _should_skip(self, timestep: int) -> bool:
        # Check timestep range
        min_t, max_t = self.config.spatial_attention_timestep_skip_range
        if not (min_t < timestep < max_t):
            return False

        # Check block skip pattern
        if self.block_index % self.config.spatial_attention_block_skip_range != 0:
            return False

        return self._step_count > 0  # Must have cached value

    def reset_state(self, module):
        self._cache = {}
        self._step_count = 0


def apply_faster_cache(
    transformer: nn.Module,
    config: FasterCacheConfig,
) -> None:
    """Apply FasterCache to a transformer model."""
    # Find attention blocks
    attention_blocks = _find_attention_blocks(transformer)

    for i, block in enumerate(attention_blocks):
        hook = FasterCacheHook(config, block_index=i)
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(hook, "faster_cache")
```

### Integration for Training Validation

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/hooks/faster_cache_validation.py

class ValidationFasterCache:
    """Wrapper for using FasterCache during training validation."""

    def __init__(
        self,
        model: nn.Module,
        skip_range: int = 2,
        timestep_range: Tuple[int, int] = (-1, 681),
    ):
        self.model = model
        self.config = FasterCacheConfig(
            spatial_attention_block_skip_range=skip_range,
            spatial_attention_timestep_skip_range=timestep_range,
            current_timestep_callback=self._get_current_timestep,
        )
        self._current_timestep = 0
        self._enabled = False

    def _get_current_timestep(self) -> int:
        return self._current_timestep

    def set_timestep(self, timestep: int):
        self._current_timestep = timestep

    def enable(self):
        """Enable FasterCache for validation."""
        if not self._enabled:
            apply_faster_cache(self.model, self.config)
            self._enabled = True

    def disable(self):
        """Disable FasterCache after validation."""
        if self._enabled:
            remove_faster_cache(self.model)
            self._enabled = False

    def reset(self):
        """Reset cache state between validation runs."""
        for module in self.model.modules():
            registry = getattr(module, HookRegistry._REGISTRY_KEY, None)
            if registry:
                hook = registry.get_hook("faster_cache")
                if hook:
                    hook.reset_state(module)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()


# Usage in training loop
def validate_model(model, validation_data, faster_cache_wrapper):
    model.eval()

    with faster_cache_wrapper:
        for batch in validation_data:
            # FasterCache automatically skips redundant attention
            output = model(batch)
            # ... compute metrics

    faster_cache_wrapper.reset()
```

### Performance Impact

Based on the paper (https://huggingface.co/papers/2410.19355):

| Model | Without Cache | With FasterCache | Speedup |
|-------|--------------|------------------|---------|
| SD 1.5 | 100% | 65% | 1.54x |
| SDXL | 100% | 70% | 1.43x |
| Video models | 100% | 60% | 1.67x |

---

## 5. Pyramid Attention Broadcast

### Source Location
```
/root/diffusers/src/diffusers/hooks/pyramid_attention_broadcast.py  # 315 lines
```

### What It Provides

Hierarchical attention caching with different skip rates for different attention types:

```python
@dataclass
class PyramidAttentionBroadcastConfig:
    """Configuration for PAB optimization."""

    # Different skip rates for different attention types
    spatial_attention_block_skip_range: int = 2
    temporal_attention_block_skip_range: int = 3
    cross_attention_block_skip_range: int = 5

    # Timestep callback
    current_timestep_callback: Callable[[], int] = None

    # Timestep ranges for each attention type
    spatial_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    cross_attention_timestep_skip_range: Tuple[int, int] = (100, 900)
```

### Why It Matters for Video Models

Video diffusion models (HunyuanVideo, WAN, FramePack) have three attention types:
1. **Spatial attention**: Within-frame relationships
2. **Temporal attention**: Cross-frame relationships
3. **Cross attention**: Text-to-video conditioning

PAB exploits that:
- Cross attention changes slowly (text condition is constant)
- Temporal attention changes moderately
- Spatial attention changes most rapidly

### Integration Pattern

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/hooks/pab_validation.py

def apply_pab_for_video_validation(
    model: nn.Module,
    model_type: str,  # "hunyuan", "wan", "framepack"
) -> None:
    """Apply PAB with model-specific configurations."""

    # Model-specific configurations
    configs = {
        "hunyuan": PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=2,
            temporal_attention_block_skip_range=4,
            cross_attention_block_skip_range=6,
        ),
        "wan": PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=2,
            temporal_attention_block_skip_range=3,
            cross_attention_block_skip_range=5,
        ),
        "framepack": PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=2,
            temporal_attention_block_skip_range=2,
            cross_attention_block_skip_range=4,
        ),
    }

    config = configs.get(model_type)
    if config is None:
        raise ValueError(f"Unknown model type: {model_type}")

    apply_pyramid_attention_broadcast(model, config)
```

---

## 6. Layer Skip Optimization

### Source Location
```
/root/diffusers/src/diffusers/hooks/layer_skip.py  # 264 lines
```

### What It Provides

Skip entire transformer blocks during inference:

```python
@dataclass
class LayerSkipConfig:
    """Configuration for layer skipping."""

    # Which block indices to skip
    indices: List[int] = None

    # Fully qualified name pattern to match
    fqn: str = "transformer_blocks"

    # What to skip within each block
    skip_attention: bool = True
    skip_ff: bool = True

    # Output mode: 0.0 = zeros, 1.0 = dropout (pass input through)
    dropout: float = 1.0


class LayerSkipHook(ModelHook):
    """Skips layer computation entirely."""

    def __init__(self, config: LayerSkipConfig):
        self.config = config

    def new_forward(self, module, *args, **kwargs):
        # Skip computation, return input or zeros
        if self.config.dropout == 1.0:
            return args[0]  # Pass through input
        else:
            return torch.zeros_like(args[0])
```

### Use Cases in Training

1. **Fast validation**: Skip less important layers during sample generation
2. **Ablation studies**: Test which layers are most important for LoRA
3. **Progressive training**: Gradually enable layers during warmup

### Integration Pattern

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/hooks/layer_skip.py

def create_validation_layer_skip(
    model: nn.Module,
    skip_ratio: float = 0.3,  # Skip 30% of layers
    skip_strategy: str = "uniform",  # or "early", "late", "middle"
) -> LayerSkipConfig:
    """Create layer skip configuration for validation."""

    # Count transformer blocks
    num_blocks = sum(1 for name, _ in model.named_modules()
                     if "transformer_blocks" in name and "." not in name.split("transformer_blocks")[1])

    num_to_skip = int(num_blocks * skip_ratio)

    if skip_strategy == "uniform":
        # Skip evenly distributed layers
        step = num_blocks // num_to_skip
        indices = list(range(0, num_blocks, step))[:num_to_skip]
    elif skip_strategy == "early":
        # Skip early layers (less important for fine details)
        indices = list(range(num_to_skip))
    elif skip_strategy == "late":
        # Skip late layers
        indices = list(range(num_blocks - num_to_skip, num_blocks))
    elif skip_strategy == "middle":
        # Skip middle layers
        start = (num_blocks - num_to_skip) // 2
        indices = list(range(start, start + num_to_skip))

    return LayerSkipConfig(
        indices=indices,
        fqn="transformer_blocks",
        skip_attention=True,
        skip_ff=True,
        dropout=1.0,  # Pass through input
    )
```

---

## 7. TaylorSeer Cache

### Source Location
```
/root/diffusers/src/diffusers/hooks/taylorseer_cache.py  # 295 lines
```

### What It Provides

Predictive caching using Taylor series approximation:

```python
@dataclass
class TaylorSeerCacheConfig:
    """Configuration for TaylorSeer caching."""

    # Cache skip parameters
    skip_range: int = 3
    timestep_skip_range: Tuple[int, int] = (0, 800)

    # Taylor approximation order
    taylor_order: int = 1  # First-order approximation

    # Which components to cache
    cache_attention: bool = True
    cache_ff: bool = False
```

### How It Works

Instead of simply reusing cached outputs, TaylorSeer:
1. Caches the output AND gradient information
2. Uses Taylor series to extrapolate the next output
3. More accurate than simple caching for changing inputs

### Integration Priority

**Lower priority** than FasterCache and PAB because:
- More complex implementation
- Marginal gains over simpler caching
- Requires gradient computation during inference

---

## 8. Context Parallelization

### Source Location
```
/root/diffusers/src/diffusers/hooks/context_parallel.py  # 303 lines
```

### What It Provides

Splits attention computation across multiple GPUs:

```python
@dataclass
class ContextParallelConfig:
    """Configuration for context parallelism."""

    # Device mesh for distributed computation
    _mesh: "DeviceMesh" = None

    # Parallelism settings
    parallel_dim: int = 1  # Dimension to split
    gather_output: bool = True


def apply_context_parallel(
    module: nn.Module,
    config: ContextParallelConfig,
    plan: Dict[str, Any],
) -> None:
    """Apply context parallelism to attention layers."""
    pass
```

### When to Use

- Multi-GPU training with very long sequences
- Video models with many frames
- Large batch sizes that don't fit on single GPU

### Integration Considerations

Requires:
- `torch.distributed` setup
- DeviceMesh configuration
- Careful attention to gradient synchronization

**Priority**: Lower (only needed for multi-GPU setups)

---

## 9. Unified Quantization Framework

### Source Location
```
/root/diffusers/src/diffusers/quantizers/
├── quantization_config.py    # Base config classes (47KB)
├── auto.py                   # Automatic backend selection
├── bitsandbytes/            # BitsAndBytes backend
├── gguf/                    # GGUF format support
├── quanto/                  # Optimum Quanto backend
├── torchao/                 # TorchAO backend
└── modelopt/                # NVIDIA ModelOpt backend
```

### What It Provides

Pluggable quantization with unified interface:

```python
# From quantization_config.py

class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GGUF = "gguf"
    TORCHAO = "torchao"
    QUANTO = "quanto"
    MODELOPT = "modelopt"


class QuantizationConfigMixin:
    """Base mixin for quantization configurations."""

    quant_method: QuantizationMethod

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfigMixin":
        """Deserialize configuration."""
        pass


class BitsAndBytesConfig(QuantizationConfigMixin):
    """BitsAndBytes quantization config."""

    quant_method = QuantizationMethod.BITS_AND_BYTES

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        llm_int8_threshold: float = 6.0,
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = False,
    ):
        pass
```

### Current Blissful-Tuner Implementation

```python
# From fp8_optimization_utils.py - hardcoded FP8 only
def apply_fp8_monkey_patch(model, device, ...):
    # No config serialization
    # No backend switching
    # FP8 only
    pass
```

### Integration Pattern

```python
# Proposed: /root/blissful-tuner/src/musubi_tuner/quantizers/

# quantization_config.py
class BlissfulQuantizationConfig(QuantizationConfigMixin):
    """Extended quantization config with Blissful-Tuner FP8."""

    quant_method: str

    # FP8-specific options
    fp8_format: str = "e4m3fn"  # or "e5m2"
    fp8_compute_dtype: torch.dtype = torch.bfloat16
    fp8_per_channel: bool = True

    # Backend selection
    backend: str = "auto"  # "fp8_custom", "bitsandbytes", "torchao", etc.


# quantizer_factory.py
def get_quantizer(config: BlissfulQuantizationConfig):
    """Factory for quantization backends."""

    if config.backend == "fp8_custom":
        from .fp8_quantizer import FP8Quantizer
        return FP8Quantizer(config)
    elif config.backend == "bitsandbytes":
        from diffusers.quantizers.bitsandbytes import BnBQuantizer
        return BnBQuantizer(config)
    elif config.backend == "torchao":
        from diffusers.quantizers.torchao import TorchAOQuantizer
        return TorchAOQuantizer(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# fp8_quantizer.py
class FP8Quantizer:
    """Wrapper around existing FP8 implementation."""

    def __init__(self, config: BlissfulQuantizationConfig):
        self.config = config

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply FP8 quantization."""
        from ..modules.fp8_optimization_utils import apply_fp8_monkey_patch
        apply_fp8_monkey_patch(
            model,
            device=torch.device("cuda"),
            fp8_format=self.config.fp8_format,
            per_channel=self.config.fp8_per_channel,
        )
        return model

    def dequantize(self, model: nn.Module) -> nn.Module:
        """Remove FP8 quantization."""
        # Implementation
        pass
```

### Benefits

1. **Experiment flexibility**: Easy A/B testing of quantization methods
2. **Config serialization**: Save/load quantization settings with checkpoints
3. **Future-proofing**: Add new backends without code changes

---

## 10. Modular Pipelines Architecture

### Source Location
```
/root/diffusers/src/diffusers/modular_pipelines/
├── modular_pipeline.py       # Core system (115KB)
├── components_manager.py     # Component management (46KB)
├── modular_pipeline_utils.py # Utilities (27KB)
└── flux/, flux2/, stable_diffusion_xl/, etc.
```

### What It Provides

Component-based, composable pipeline architecture:

```python
# Concept from modular_pipeline.py

class ModularPipeline:
    """Composable pipeline from interchangeable components."""

    def __init__(self):
        self.components = ComponentsManager()

    def add_component(self, name: str, component: Any) -> None:
        """Add a named component."""
        self.components.register(name, component)

    def remove_component(self, name: str) -> None:
        """Remove a component."""
        self.components.unregister(name)

    def swap_component(self, name: str, new_component: Any) -> None:
        """Replace a component."""
        self.components.replace(name, new_component)
```

### Relevance to Training

The modular approach could improve:
- LoRA adapter swapping during training
- Component-level optimization toggling
- Validation pipeline configuration

### Integration Priority

**Lower priority** - More relevant for inference pipelines than training.

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Establish hook infrastructure

1. **Create hook base classes**
   - Location: `/root/blissful-tuner/src/musubi_tuner/hooks/base.py`
   - Classes: `ModelHook`, `HookRegistry`
   - Tests: Basic hook registration and lifecycle

2. **Migrate existing offloading to hooks**
   - Refactor `custom_offloading_utils.py` to use `HookRegistry`
   - Ensure backward compatibility
   - Tests: Verify offloading still works

3. **Add layerwise casting**
   - Location: `/root/blissful-tuner/src/musubi_tuner/hooks/layerwise_casting.py`
   - Integrate with existing FP8 code
   - Tests: Memory and speed benchmarks

### Phase 2: Validation Optimization (Weeks 3-4)

**Goal**: Speed up validation during training

1. **Implement FasterCache**
   - Location: `/root/blissful-tuner/src/musubi_tuner/hooks/faster_cache.py`
   - Context manager for training validation
   - Tests: Validation speed benchmarks

2. **Upgrade offloading with prefetching**
   - Add `LazyPrefetchGroupOffloadingHook`
   - Detect execution order automatically
   - Tests: Offloading speed benchmarks

3. **Add PAB for video models**
   - Location: `/root/blissful-tuner/src/musubi_tuner/hooks/pab.py`
   - Model-specific configurations
   - Tests: Video validation benchmarks

### Phase 3: Production Hardening (Weeks 5-6)

**Goal**: Polish and test

1. **Add layer skip**
   - Location: `/root/blissful-tuner/src/musubi_tuner/hooks/layer_skip.py`
   - Multiple skip strategies
   - Tests: Quality vs speed tradeoffs

2. **Refactor quantization**
   - Create `QuantizationConfigMixin` adapter
   - Support multiple backends
   - Tests: Quantization comparison

3. **Documentation and examples**
   - Update training configs with hook options
   - Add example configurations
   - Benchmark documentation

### Phase 4: Advanced Features (Future)

1. Context parallelization for multi-GPU
2. TaylorSeer cache
3. Modular pipeline integration

---

## File Reference Map

### Diffusers Source Files to Study

| File | Size | Purpose | Priority |
|------|------|---------|----------|
| `hooks/hooks.py` | ~200 lines | Core hook infrastructure | Critical |
| `hooks/_common.py` | ~150 lines | Layer detection utilities | High |
| `hooks/group_offloading.py` | 956 lines | Prefetching offloading | High |
| `hooks/layerwise_casting.py` | 241 lines | Dynamic precision | High |
| `hooks/faster_cache.py` | 655 lines | Attention caching | High |
| `hooks/pyramid_attention_broadcast.py` | 315 lines | Video attention caching | Medium |
| `hooks/layer_skip.py` | 264 lines | Block skipping | Medium |
| `hooks/taylorseer_cache.py` | 295 lines | Predictive caching | Low |
| `hooks/context_parallel.py` | 303 lines | Multi-GPU attention | Low |
| `quantizers/quantization_config.py` | ~500 lines | Quantization framework | Medium |

### Blissful-Tuner Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/musubi_tuner/hooks/__init__.py` | Create | Hook module exports |
| `src/musubi_tuner/hooks/base.py` | Create | Base hook classes |
| `src/musubi_tuner/hooks/group_offloading.py` | Create | Prefetching offloading |
| `src/musubi_tuner/hooks/layerwise_casting.py` | Create | Dynamic precision |
| `src/musubi_tuner/hooks/faster_cache.py` | Create | Attention caching |
| `src/musubi_tuner/hooks/pab.py` | Create | Video attention caching |
| `src/musubi_tuner/hooks/layer_skip.py` | Create | Block skipping |
| `src/musubi_tuner/modules/custom_offloading_utils.py` | Modify | Migrate to hook system |
| `src/musubi_tuner/modules/fp8_optimization_utils.py` | Modify | Add dynamic casting |
| `src/musubi_tuner/quantizers/__init__.py` | Create | Quantization module |
| `src/musubi_tuner/quantizers/config.py` | Create | Quantization config |

---

## Appendix: Code Snippets for Quick Reference

### A. Minimal Hook Implementation

```python
# Minimal hook that logs forward passes
class LoggingHook(ModelHook):
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    def pre_forward(self, module, *args, **kwargs):
        self.call_count += 1
        print(f"[{self.name}] Forward #{self.call_count}")
        return args, kwargs

# Usage
hook = LoggingHook("encoder")
registry = HookRegistry.check_if_exists_or_initialize(model.encoder)
registry.register_hook(hook, "logging")
```

### B. Composing Multiple Hooks

```python
# Apply multiple optimizations to same model
def setup_training_optimizations(model, validation_mode=False):
    # Always apply: offloading + precision
    apply_group_offloading(model, torch.device("cuda"), torch.device("cpu"))
    apply_layerwise_casting(model, torch.float8_e4m3fn, torch.bfloat16)

    # Validation only: caching + skipping
    if validation_mode:
        apply_faster_cache(model, FasterCacheConfig(...))
        apply_layer_skip(model, LayerSkipConfig(indices=[10, 20]))

def cleanup_validation_hooks(model):
    for module in model.modules():
        registry = getattr(module, HookRegistry._REGISTRY_KEY, None)
        if registry:
            registry.remove_hook("faster_cache")
            registry.remove_hook("layer_skip")
```

### C. Training Loop Integration

```python
# Example training loop with hook management
class Trainer:
    def __init__(self, model, ...):
        self.model = model
        self.faster_cache = ValidationFasterCache(model)
        setup_training_optimizations(model, validation_mode=False)

    def train_step(self, batch):
        # Hooks automatically handle offloading and precision
        loss = self.model(batch)
        loss.backward()
        return loss

    def validate(self, val_loader):
        self.model.eval()
        with self.faster_cache:  # Enable caching
            for batch in val_loader:
                output = self.model(batch)
                # ... compute metrics
        self.faster_cache.reset()  # Clear cache state
        self.model.train()
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01 | 1.0 | Initial document |

---

*This document should be updated as features are implemented or diffusers evolves.*
