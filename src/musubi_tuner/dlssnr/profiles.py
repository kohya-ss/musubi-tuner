# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Fixed DLSS-NR 310.8.0 record layout.

Byte ranges come from the OpenDLSS fused-layout rules and were checked against
the shipped manifest: 153 records, 147,683,778 payload bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from musubi_tuner.dlssnr.packing import align_up

PROFILE_ID = "dlss_nr_310_8_0"

STAGE_BYTES = {
    "enc32": 106_432,
    "enc64": 255_216,
    "enc128": 1_215_856,
    "enc256": 5_644_912,
    "enc512": 16_269_840,
    "vit": 100_697_232,
    "dec512": 16_270_848,
    "dec256": 5_645_408,
    "dec128": 1_216_096,
    "dec64": 255_328,
    "dec32": 106_610,
}

# U1 measurements used to identify this exact checkpoint. They are not training clamps.
SOURCE_FINGERPRINT = {
    "e4_count": 143_831_040,
    "e4_max_abs": 0.875,
    "e4_pos_zero": 3_617_037,
    "e4_neg_zero": 3_617_790,
    "e4_abs_ge_0_125": 1_852_806,
    "e4_abs_ge_0_5": 1_968,
    "e4_nan": 0,
    "from_vit_max_abs": 0.140625,
    "prior_count": 1_875_968,
    "prior_min": -126.5,
    "prior_max": 0.0,
    "prior_zeros": 200,
    "temperature_count": 714,
    "temperature_min": 0.0145336,
    "temperature_max": 28.4389,
    "pad_bytes": 2_376,
    "head_unused_halves": 384,
    "blend_scale": 0.73974609375,
    "logical_parameters_without_blend": 145_755_114,
}


@dataclass(frozen=True)
class View:
    """One canonical tensor cut out of a packed storage region. Ranges are half-open on [K, N]."""

    name: str
    group: str
    k0: int
    k1: int
    n0: int
    n1: int

    @property
    def canonical_shape(self) -> tuple[int, ...]:
        return (self.n1 - self.n0, self.k1 - self.k0)


@dataclass(frozen=True)
class Region:
    kind: str  # e4, f16, f32, f16frag, prior, pad, opaque
    offset: int
    nbytes: int
    k: int = 0
    n: int = 0
    heads: int = 0
    views: tuple[View, ...] = ()

    def logical_count(self) -> int:
        if self.kind in ("pad", "opaque"):
            return 0
        if self.kind == "prior":
            return self.heads * 64 * 64
        if self.kind in ("f16", "f32"):
            return self.nbytes // (2 if self.kind == "f16" else 4)
        return sum((view.k1 - view.k0) * (view.n1 - view.n0) for view in self.views)


@dataclass(frozen=True)
class RecordLayout:
    name: str
    block: int
    layer: int
    parameter: str
    stage: str
    phase: int | None
    regions: tuple[Region, ...]

    @property
    def nbytes(self) -> int:
        return sum(region.nbytes for region in self.regions)


def _view(name: str, group: str, k: int, n: int, k0: int = 0, n0: int = 0) -> View:
    return View(name, group, k0, k0 + k, n0, n0 + n)


def _full(name: str, group: str, k: int, n: int) -> tuple[View, ...]:
    return (_view(name, group, k, n),)


class _Builder:
    def __init__(self) -> None:
        self.regions: list[Region] = []
        self.offset = 0

    def add(self, kind: str, nbytes: int, **kwargs) -> None:
        self.regions.append(Region(kind=kind, offset=self.offset, nbytes=nbytes, **kwargs))
        self.offset += nbytes

    def e4(self, name_or_views, k: int, n: int, group: str = "matrices") -> None:
        views = name_or_views if isinstance(name_or_views, tuple) else _full(name_or_views, group, k, n)
        self.add("e4", k * n, k=k, n=n, views=views)

    def f16_vec(self, name: str, count: int, group: str) -> None:
        self.add("f16", count * 2, k=count, n=1, views=_full(name, group, count, 1))

    def f32_vec(self, name: str, count: int, group: str = "scales") -> None:
        width = count * 4
        padded = align_up(width, 16)
        self.add("f32", width, k=count, n=1, views=_full(name, group, count, 1))
        if padded != width:
            self.add("pad", padded - width)

    def prior(self, name: str, heads: int) -> None:
        self.add("prior", heads * 8192, heads=heads, views=_full(name, "priors", heads * 64 * 64, 1))

    def f16_matrix(self, views: tuple[View, ...], k: int, n: int) -> None:
        n_tiles = (n + 15) // 16
        k_tiles = (k + 15) // 16
        self.add("f16frag", k_tiles * n_tiles * 512, k=k, n=n, views=views)

    def pad(self, nbytes: int = 16) -> None:
        self.add("pad", nbytes)

    def opaque(self, nbytes: int) -> None:
        self.add("opaque", nbytes)


def _prefix(block: int) -> str:
    return f"blocks.{block}"


def _expert_views(block: int, stem: str, experts: int, k_each: int, n: int, group: str = "matrices") -> tuple[View, ...]:
    views = []
    for expert in range(experts):
        views.append(
            View(
                f"{_prefix(block)}.ffn.experts.{expert}.{stem}",
                group,
                expert * k_each,
                (expert + 1) * k_each,
                0,
                n,
            )
        )
    return tuple(views)


def _column_views(block: int, stem: str, branches: int, k: int, n_each: int) -> tuple[View, ...]:
    views = []
    for branch in range(branches):
        views.append(
            View(
                f"{_prefix(block)}.ffn.branches.{branch}.{stem}",
                "matrices",
                0,
                k,
                branch * n_each,
                (branch + 1) * n_each,
            )
        )
    return tuple(views)


def _row_views(block: int, stem: str, branches: int, k_each: int, n: int) -> tuple[View, ...]:
    views = []
    for branch in range(branches):
        views.append(
            View(
                f"{_prefix(block)}.ffn.branches.{branch}.{stem}",
                "matrices",
                branch * k_each,
                (branch + 1) * k_each,
                0,
                n,
            )
        )
    return tuple(views)


def _ffn_weights(builder: _Builder, block: int, channels: int) -> None:
    if channels == 32:
        builder.e4(f"{_prefix(block)}.ffn.fc1.weight", 32, 128)
        builder.e4(f"{_prefix(block)}.ffn.fc2.weight", 128, 32)
        return
    if channels in (64, 128, 256):
        experts = channels // 32
        builder.e4(_expert_views(block, "fc1.weight", experts, channels, 128), experts * channels, 128)
        builder.e4(_expert_views(block, "fc2.weight", experts, 128, 32), experts * 128, 32)
        builder.e4(f"{_prefix(block)}.ffn.fc3.weight", channels, channels)
        return
    raise ValueError(f"no dense/expert FFN layout for {channels} channels")


def _attention(builder: _Builder, block: int, channels: int) -> None:
    heads = channels // 32
    builder.e4(f"{_prefix(block)}.attn.qkv.weight", channels, channels * 3)
    builder.prior(f"{_prefix(block)}.attn.prior", heads)
    builder.f32_vec(f"{_prefix(block)}.attn.temperature", heads)
    builder.e4(f"{_prefix(block)}.attn.proj.weight", channels, channels)
    builder.f16_vec(f"{_prefix(block)}.attn.skip_scale", channels, "scales")


def _fused(block: int, channels: int, transition: tuple[int, int] | None, tail_pad: bool) -> _Builder:
    builder = _Builder()
    _ffn_weights(builder, block, channels)
    builder.pad()
    builder.f16_vec(f"{_prefix(block)}.ffn.skip_scale", channels, "scales")
    builder.pad()
    _attention(builder, block, channels)
    if transition is not None:
        k_size, n_size = transition
        builder.e4(f"{_prefix(block)}.down.weight", k_size, n_size)
        if tail_pad:
            builder.pad()
    elif tail_pad:
        builder.pad()
    return builder


def _pre(block: int = 0) -> _Builder:
    builder = _Builder()
    builder.e4(f"{_prefix(block)}.ffn.fc1.weight", 32, 128)
    builder.e4(f"{_prefix(block)}.ffn.fc2.weight", 128, 32)
    builder.pad()
    builder.f16_matrix(_full(f"{_prefix(block)}.input_adapter.weight", "input_rgb_head", 16, 32), 16, 32)
    builder.f16_vec(f"{_prefix(block)}.ffn.skip_scale", 32, "scales")
    builder.pad()
    _attention(builder, block, 32)
    builder.pad()
    return builder


def _post(block: int = 70) -> _Builder:
    builder = _Builder()
    builder.e4(f"{_prefix(block)}.ffn.fc1.weight", 32, 128)
    builder.e4(f"{_prefix(block)}.ffn.fc2.weight", 128, 32)
    builder.pad()
    builder.f16_vec(f"{_prefix(block)}.ffn.skip_scale", 32, "scales")
    builder.f16_vec(f"{_prefix(block)}.merge.up_scale", 32, "scales")
    builder.f16_vec(f"{_prefix(block)}.merge.adapter_scale", 32, "scales")
    builder.e4(f"{_prefix(block)}.attn.qkv.weight", 32, 96)
    builder.prior(f"{_prefix(block)}.attn.prior", 1)
    builder.f32_vec(f"{_prefix(block)}.attn.temperature", 1)
    builder.e4(f"{_prefix(block)}.attn.proj.weight", 32, 32)
    builder.f16_vec(f"{_prefix(block)}.attn.skip_scale", 32, "scales")
    builder.pad()
    head = (
        View(f"{_prefix(block)}.head.rgb.weight", "input_rgb_head", 0, 32, 0, 3),
        View(f"{_prefix(block)}.head.logit.weight", "temporal_head", 0, 32, 3, 4),
    )
    builder.f16_matrix(head, 32, 4)
    return builder


def _upsample(block: int, channels: int) -> _Builder:
    builder = _Builder()
    _ffn_weights(builder, block, channels)
    builder.e4(f"{_prefix(block)}.up.weight", channels * 2, channels)
    if channels == 32:
        builder.pad()
    builder.f16_vec(f"{_prefix(block)}.ffn.skip_scale", channels, "scales")
    if channels == 32:
        builder.pad()
    builder.f16_vec(f"{_prefix(block)}.up.skip_scale", channels, "scales")
    _attention(builder, block, channels)
    builder.pad()
    return builder


def _split512(block: int) -> list[tuple[int, _Builder]]:
    layers: list[tuple[int, _Builder]] = []
    layer0 = _Builder()
    layer0.e4(_column_views(block, "in_proj.weight", 8, 512, 64), 512, 512)
    layer0.e4(_row_views(block, "fc1.weight", 8, 64, 256), 8 * 64, 256)
    layer0.e4(_row_views(block, "fc2.weight", 8, 256, 64), 8 * 256, 64)
    layers.append((0, layer0))

    layer1 = _Builder()
    layer1.e4(f"{_prefix(block)}.ffn.contract.weight", 512, 512)
    layer1.f16_vec(f"{_prefix(block)}.ffn.skip_scale", 512, "scales")
    layers.append((1, layer1))

    layer2 = _Builder()
    layer2.e4(f"{_prefix(block)}.attn.qkv.weight", 512, 1536)
    layer2.prior(f"{_prefix(block)}.attn.prior", 16)
    layer2.f32_vec(f"{_prefix(block)}.attn.temperature", 16)
    layers.append((2, layer2))

    layer3 = _Builder()
    layer3.e4(f"{_prefix(block)}.attn.proj.weight", 512, 512)
    layer3.f16_vec(f"{_prefix(block)}.attn.skip_scale", 512, "scales")
    layers.append((3, layer3))
    return layers


def _vit(block: int) -> list[tuple[int, _Builder]]:
    layers: list[tuple[int, _Builder]] = []
    layer0 = _Builder()
    layer0.e4(f"{_prefix(block)}.ffn.fc1.weight", 1024, 4096)
    layer0.pad()
    layers.append((0, layer0))

    layer1 = _Builder()
    layer1.e4(f"{_prefix(block)}.ffn.fc2.weight", 4096, 1024)
    layer1.f16_vec(f"{_prefix(block)}.ffn.skip_scale", 1024, "scales")
    layers.append((1, layer1))

    layer2 = _Builder()
    layer2.f32_vec(f"{_prefix(block)}.attn.temperature", 32)
    layer2.e4(f"{_prefix(block)}.attn.qkv.weight", 1024, 3072)
    layers.append((2, layer2))

    layer3 = _Builder()
    layer3.opaque(2)
    layers.append((3, layer3))

    layer4 = _Builder()
    layer4.e4(f"{_prefix(block)}.attn.proj.weight", 1024, 1024)
    layer4.f16_vec(f"{_prefix(block)}.attn.skip_scale", 1024, "scales")
    layers.append((4, layer4))
    return layers


def _phase_table() -> dict[int, int | None]:
    """Window phase consumed in graph order. ViT and block 39 do not take one."""
    counters = [0] * 7
    phases: dict[int, int | None] = {}

    def take(block: int, level: int) -> None:
        phases[block] = counters[level] & 3
        counters[level] += 1

    take(0, 6)
    for block in range(1, 5):
        take(block, 0)
    for block in range(5, 9):
        take(block, 1)
    for block in range(9, 15):
        take(block, 2)
    for block in range(15, 23):
        take(block, 3)
    for block in range(23, 31):
        take(block, 4)
    for block in range(31, 40):
        phases[block] = None
    for block in range(40, 48):
        take(block, 4)
    for block in range(48, 56):
        take(block, 3)
    for block in range(56, 62):
        take(block, 2)
    for block in range(62, 66):
        take(block, 1)
    for block in range(66, 70):
        take(block, 0)
    take(70, 6)
    return phases


def _stage_for(block: int) -> str:
    if block <= 4:
        return "enc32"
    if block <= 8:
        return "enc64"
    if block <= 14:
        return "enc128"
    if block <= 22:
        return "enc256"
    if block <= 30:
        return "enc512"
    if block <= 38:
        return "vit"
    if block <= 47:
        return "dec512"
    if block <= 55:
        return "dec256"
    if block <= 61:
        return "dec128"
    if block <= 65:
        return "dec64"
    return "dec32"


def _record(name_block: int, layer: int, parameter: str, builder: _Builder, phases: dict[int, int | None]) -> RecordLayout:
    return RecordLayout(
        name=f"block{name_block}.layer{layer}.{parameter}",
        block=name_block,
        layer=layer,
        parameter=parameter,
        stage=_stage_for(name_block),
        phase=phases[name_block],
        regions=tuple(builder.regions),
    )


def build_records() -> tuple[RecordLayout, ...]:
    phases = _phase_table()
    records: list[RecordLayout] = []
    records.append(_record(0, 0, "layer", _pre(), phases))

    normal = [(range(1, 5), 32, 4), (range(5, 9), 64, 8), (range(9, 15), 128, 14), (range(15, 23), 256, 22)]
    for blocks, channels, last in normal:
        for block in blocks:
            is_last = block == last
            transition = (channels, channels * 2) if is_last else None
            tail = (not is_last) or channels == 32
            records.append(_record(block, 0, "layer", _fused(block, channels, transition, tail), phases))

    for block in range(23, 31):
        for layer, builder in _split512(block):
            records.append(_record(block, layer, "layer", builder, phases))
        if block == 30:
            extra = _Builder()
            extra.e4(f"{_prefix(block)}.to_vit.weight", 512, 1024)
            extra.pad()
            records.append(_record(block, 4, "layer", extra, phases))

    for block in range(31, 39):
        for layer, builder in _vit(block):
            records.append(_record(block, layer, "layer", builder, phases))

    bridge = _Builder()
    bridge.e4("blocks.39.proj.weight", 1024, 512)
    bridge.f16_vec("blocks.39.skip_scale", 512, "scales")
    records.append(_record(39, 0, "layer", bridge, phases))

    for block in range(40, 48):
        for layer, builder in _split512(block):
            records.append(_record(block, layer, "layer", builder, phases))

    decoder = [(48, 256, range(48, 56)), (56, 128, range(56, 62)), (62, 64, range(62, 66)), (66, 32, range(66, 70))]
    for first, channels, blocks in decoder:
        for block in blocks:
            builder = _upsample(block, channels) if block == first else _fused(block, channels, None, True)
            records.append(_record(block, 0, "layer", builder, phases))

    blend = _Builder()
    blend.f16_vec("blocks.70.blend_scale", 1, "temporal_blend")
    records.append(_record(70, 0, "blend_scale", blend, phases))
    records.append(_record(70, 0, "layer", _post(), phases))
    return tuple(records)


def block_channels(block: int) -> int | None:
    if block == 39:
        return None
    if block <= 4 or block >= 66:
        return 32
    if block <= 8 or block >= 62:
        return 64
    if block <= 14 or block >= 56:
        return 128
    if block <= 22 or block >= 48:
        return 256
    if block <= 30 or block >= 40:
        return 512
    return 1024


def block_kind(block: int) -> str:
    if block == 39:
        return "transition"
    if 31 <= block <= 38:
        return "vit"
    if 23 <= block <= 30 or 40 <= block <= 47:
        return "split512"
    if block in (0, 70):
        return "window32_boundary"
    return "window"


def canonical_names(records: tuple[RecordLayout, ...] | None = None) -> list[str]:
    records = records if records is not None else build_records()
    names: list[str] = []
    for record in records:
        for region in record.regions:
            for view in region.views:
                names.append(view.name)
    if len(names) != len(set(names)):
        raise RuntimeError("canonical tensor names are not unique")
    return names
