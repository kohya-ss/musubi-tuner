"""A generic composer for per-scheme prequantized-checkpoint loaders.

Some checkpoints declare different quantization schemes on different modules within the
same file (e.g. NVFP4 on some Linears, ConvRot INT8 on others). Musubi's per-scheme
loaders (`NvFp4Quantizer`, `ConvRotInt8Quantizer`, and any future one) each convert their
own scheme's tensors into Musubi's internal layout as soon as they read them, and raise
on any module declared under a scheme they don't own -- unless constructed with
`foreign_formats` naming the scheme(s) owned by their peers, in which case they silently
skip those modules instead.

`MixedQuantizer` has no awareness of any specific scheme or source format: it just runs
each already-configured sub-loader over the same files and merges their output state
dicts. Every module is converted by exactly the sub-loader that owns it, so the merge is
lossless. Adding a new co-resident scheme means constructing its own loader with
`foreign_formats` covering its peers and adding one entry to the dict passed into this
class -- the class itself never changes.
"""

from typing import Dict, List, Optional, Union

import torch

from musubi_tuner.utils.safetensors_utils import WeightTransformHooks


class MixedQuantizer:
    """Composes named, already-configured per-format quantizers into one loader.

    ``quantizers`` maps an arbitrary caller-chosen name (e.g. ``"nvfp4"``,
    ``"convrot_int8"``) to a quantizer instance implementing the same
    ``load_and_quantize`` protocol as ``NvFp4Quantizer`` / ``ConvRotInt8Quantizer``. Each
    quantizer must already be constructed with ``foreign_formats`` covering every *other*
    entry's owned format(s), so it skips (rather than raises on) modules it doesn't own.

    After ``load_and_quantize`` returns, reach into ``self.quantizers[name]`` for that
    sub-quantizer's own format-specific results (e.g.
    ``mixed.quantizers["nvfp4"].nvfp4_module_shapes``,
    ``mixed.quantizers["convrot_int8"].module_groupsizes``) to drive the matching
    ``apply_*_monkey_patch`` call.
    """

    def __init__(self, quantizers: Dict[str, object]):
        if not quantizers:
            raise ValueError("MixedQuantizer requires at least one sub-quantizer")
        self.quantizers = quantizers

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> Dict[str, torch.Tensor]:
        if weight_hook is not None:
            raise ValueError(
                "Cannot merge LoRA weights into a mixed-format prequantized checkpoint."
                " Use the original BF16 weights to merge LoRA at load time, or apply the LoRA at runtime."
                " / 混在フォーマットの事前量子化済みチェックポイントにはLoRAをマージできません。"
                "BF16の元重みを使用してロード時マージするか、LoRAを実行時適用してください。"
            )
        state_dict: Dict[str, torch.Tensor] = {}
        for quantizer in self.quantizers.values():
            sub_state_dict = quantizer.load_and_quantize(
                model_files,
                calc_device,
                move_to_device=move_to_device,
                disable_numpy_memmap=disable_numpy_memmap,
                weight_transform_hooks=weight_transform_hooks,
            )
            state_dict.update(sub_state_dict)
        return state_dict
