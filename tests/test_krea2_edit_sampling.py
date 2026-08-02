from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamBlock, SingleStreamDiT
from musubi_tuner.krea2.krea2_utils import normalize_krea2_lora_state_dict, validate_krea2_lora_state_dict
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.krea2.krea2_sampling import (
    append_reference_latents,
    encode_reference_images,
    pack_reference_latents,
    sample,
)


def test_pack_reference_latents_assigns_distinct_rope_indices():
    refs = [torch.ones(1, 4, 4), torch.full((1, 2, 4), 2.0)]

    tokens, pos = pack_reference_latents(refs, batch_size=2, patch=2, device="cpu", dtype=torch.float32)

    assert tokens.shape == (2, 6, 4)
    assert torch.equal(pos[0, :4, 0], torch.ones(4))
    assert torch.equal(pos[0, 4:, 0], torch.full((2,), 2.0))
    assert torch.equal(tokens[0], tokens[1])


def test_ai_toolkit_lora_keys_are_converted_to_musubi_format():
    state_dict = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.randn(4, 16),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.randn(16, 4),
    }

    converted = normalize_krea2_lora_state_dict(state_dict)

    assert set(converted) == {
        "lora_unet_blocks_0_attn_wq.lora_down.weight",
        "lora_unet_blocks_0_attn_wq.lora_up.weight",
    }


def test_krea2_lora_validation_rejects_zero_matching_layers():
    model = SingleStreamDiT(
        SingleMMDiTConfig(
            features=16,
            tdim=8,
            txtdim=16,
            heads=1,
            multiplier=1,
            layers=1,
            patch=1,
            channels=1,
            txtlayers=2,
            txtheads=1,
            txtkvheads=1,
        )
    )
    state_dict = {
        "lora_unet_not_a_real_layer.lora_down.weight": torch.randn(4, 16),
        "lora_unet_not_a_real_layer.lora_up.weight": torch.randn(16, 4),
    }

    try:
        validate_krea2_lora_state_dict(model, state_dict, 0)
    except ValueError as error:
        assert "matched 0 model layers" in str(error)
    else:
        raise AssertionError("Expected a zero-match Krea 2 LoRA to be rejected")


def test_ai_toolkit_lora_is_numerically_merged(tmp_path):
    model_path = tmp_path / "model.safetensors"
    base_weight = torch.zeros(16, 16)
    save_file({"blocks.0.attn.wq.weight": base_weight}, model_path)
    down = torch.randn(4, 16)
    up = torch.randn(16, 4)
    lora = normalize_krea2_lora_state_dict(
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": down,
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": up,
        }
    )

    merged = load_safetensors_with_lora_and_fp8(
        str(model_path),
        [lora],
        [1.0],
        fp8_optimization=False,
        calc_device=torch.device("cpu"),
        dit_weight_dtype=torch.float32,
    )

    assert torch.allclose(merged["blocks.0.attn.wq.weight"], up @ down)


def test_append_reference_latents_places_refs_before_text():
    img = torch.zeros(1, 3, 4)
    pos = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 0], [0, 0, 0]]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, False]])

    combined, combined_pos, combined_mask, reflen = append_reference_latents(img, pos, mask, [torch.ones(1, 2, 2)], patch=2)

    assert combined.shape == (1, 4, 4)
    assert reflen == 1
    assert combined_pos[0, 3, 0] == 1  # first reference uses RoPE axis-0 index 1
    assert torch.equal(combined_pos[:, 4:], pos[:, 3:])
    assert torch.equal(combined_mask, torch.tensor([[True, True, True, True, True, False]]))


class _FakeEncodingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.encoded_pixels = None

    def encode_pixels_to_latents(self, pixels, *, sample_posterior=False, generator=None):
        self.encoded_pixels = pixels
        self.sample_posterior = sample_posterior
        return torch.zeros(pixels.shape[0], 1, 1, pixels.shape[-2], pixels.shape[-1], device=pixels.device)


def test_encode_reference_images_normalizes_pixels_and_snaps_size():
    vae = _FakeEncodingVAE()

    latents = encode_reference_images(
        vae,
        [torch.zeros(3, 17, 31)],
        device="cpu",
        dtype=torch.float32,
        max_pixels=1024 * 1024,
    )

    assert vae.encoded_pixels.shape == (1, 3, 16, 32)
    assert vae.encoded_pixels.min() == -1
    assert vae.encoded_pixels.max() == -1
    assert vae.sample_posterior is True
    assert latents[0].shape == (1, 16, 32)


class _GateOnlyModulation(torch.nn.Module):
    def forward(self, vec):
        zeros = torch.zeros_like(vec)
        return zeros, zeros, vec, zeros, zeros, zeros


class _IdentityAttention(torch.nn.Module):
    def forward(self, x, freqs=None, attn_params=None):
        return x


class _Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_single_stream_block_uses_t0_modulation_only_for_reference_span():
    block = SingleStreamBlock(features=16, heads=1, multiplier=1)
    block.mod = _GateOnlyModulation()
    block.prenorm = torch.nn.Identity()
    block.postnorm = torch.nn.Identity()
    block.attn = _IdentityAttention()
    block.mlp = _Zero()

    hidden = torch.ones(1, 4, 16)
    tvec = torch.ones(1, 1, 16)
    t0vec = torch.full((1, 1, 16), 2.0)

    output = block(hidden, (tvec, t0vec, 1, 3), freqs=None)

    assert torch.equal(output[0, :, 0], torch.tensor([2.0, 3.0, 3.0, 2.0]))


def test_single_stream_dit_excludes_reference_tokens_from_prediction():
    config = SingleMMDiTConfig(
        features=16,
        tdim=8,
        txtdim=16,
        heads=1,
        multiplier=1,
        layers=1,
        patch=1,
        channels=1,
        txtlayers=2,
        txtheads=1,
        txtkvheads=1,
    )
    model = SingleStreamDiT(config).eval()
    image_and_refs = torch.randn(1, 8, 1)
    context = torch.randn(1, 2, 2, 16)
    pos = torch.zeros(1, 10, 3)
    mask = torch.ones(1, 10, dtype=torch.bool)

    output = model(image_and_refs, context, torch.tensor([0.5]), pos, mask, reflen=4)

    assert output.shape == (1, 4, 1)
    assert torch.isfinite(output).all()


def test_isolated_reference_rows_do_not_depend_on_live_tokens():
    torch.manual_seed(11)
    block = SingleStreamBlock(features=16, heads=1, multiplier=1).eval()
    first = torch.randn(1, 5, 16)
    second = first.clone()
    second[:, :1] = torch.randn_like(second[:, :1])
    second[:, 3:] = torch.randn_like(second[:, 3:])
    tvec = torch.randn(1, 1, 96)
    t0vec = torch.randn(1, 1, 96)

    with torch.no_grad():
        first_out = block(first, (tvec, t0vec, 1, 3), freqs=None, ref_span=(1, 3))
        second_out = block(second, (tvec, t0vec, 1, 3), freqs=None, ref_span=(1, 3))

    assert torch.allclose(first_out[:, 1:3], second_out[:, 1:3], atol=1e-6, rtol=1e-6)


def test_reference_kv_cache_matches_joint_isolated_forward():
    torch.manual_seed(17)
    config = SingleMMDiTConfig(
        features=16,
        tdim=8,
        txtdim=16,
        heads=1,
        multiplier=1,
        layers=2,
        patch=1,
        channels=1,
        txtlayers=2,
        txtheads=1,
        txtkvheads=1,
    )
    model = SingleStreamDiT(config, attn_mode="torch").eval()
    target = torch.randn(1, 4, 1)
    refs = torch.randn(1, 3, 1)
    context = torch.randn(1, 2, 2, 16)
    target_pos = torch.zeros(1, 4, 3)
    ref_pos = torch.zeros(1, 3, 3)
    ref_pos[..., 0] = 1
    text_pos = torch.zeros(1, 2, 3)
    full_pos = torch.cat((target_pos, ref_pos, text_pos), dim=1)
    base_pos = torch.cat((target_pos, text_pos), dim=1)
    full_mask = torch.ones(1, 9, dtype=torch.bool)
    base_mask = torch.ones(1, 6, dtype=torch.bool)
    timestep = torch.tensor([0.6])
    capture = []

    with torch.no_grad():
        joint = model(
            torch.cat((target, refs), dim=1),
            context,
            timestep,
            full_pos,
            full_mask,
            reflen=3,
            isolate_refs=True,
            ref_kv_capture=capture,
        )
        cached = model(
            target,
            context,
            timestep,
            base_pos,
            base_mask,
            ref_kv_cache=capture,
        )

    assert len(capture) == config.layers
    assert torch.allclose(joint, cached, atol=2e-5, rtol=2e-5)


class _FakeEditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=1, channels=1)
        self.calls = []

    def forward(self, img, context, t, pos, mask, reflen=0):
        self.calls.append((img.shape[1], reflen, pos.shape[1], mask.shape[1]))
        return torch.zeros_like(img[:, : img.shape[1] - reflen])


class _FakeCachedEditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=1, channels=1)
        self.blocks = [None, None]
        self.calls = []

    def forward(
        self,
        img,
        context,
        t,
        pos,
        mask,
        reflen=0,
        isolate_refs=False,
        ref_kv_capture=None,
        ref_kv_cache=None,
    ):
        self.calls.append(
            {
                "imglen": img.shape[1],
                "reflen": reflen,
                "poslen": pos.shape[1],
                "isolate": isolate_refs,
                "capture": ref_kv_capture is not None,
                "cached": ref_kv_cache is not None,
            }
        )
        if ref_kv_capture is not None:
            for _ in self.blocks:
                ref_kv_capture.append((torch.zeros(1, 1, reflen, 1), torch.zeros(1, 1, reflen, 1)))
        return torch.zeros_like(img[:, : img.shape[1] - reflen])


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperal_downsample = []
        self.z_dim = 1
        self.dtype = torch.float32

    def decode_to_pixels(self, latents):
        return torch.zeros(latents.shape[0], 3, latents.shape[-2], latents.shape[-1])


def test_sample_appends_clean_refs_but_integrates_only_target_tokens():
    model = _FakeEditModel()
    vae = _FakeVAE()
    txt = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    txtmask = torch.ones(1, 1, dtype=torch.bool)

    images = sample(
        model,
        vae,
        txt,
        txtmask,
        device="cpu",
        dtype=torch.bfloat16,
        width=2,
        height=2,
        steps=1,
        cfg_scale=1.0,
        ref_latents=[torch.ones(1, 2, 2, dtype=torch.bfloat16)],
    )

    assert len(images) == 1
    # Four target tokens + four clean reference tokens; positions/mask also include one text token.
    assert model.calls == [(8, 4, 9, 9)]


def test_sample_captures_reference_kv_once_then_reuses_it():
    model = _FakeCachedEditModel()
    vae = _FakeVAE()
    txt = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    txtmask = torch.ones(1, 1, dtype=torch.bool)

    sample(
        model,
        vae,
        txt,
        txtmask,
        device="cpu",
        dtype=torch.bfloat16,
        width=2,
        height=2,
        steps=2,
        cfg_scale=1.0,
        ref_latents=[torch.ones(1, 2, 2, dtype=torch.bfloat16)],
        kv_cache=True,
    )

    assert model.calls == [
        {"imglen": 8, "reflen": 4, "poslen": 9, "isolate": True, "capture": True, "cached": False},
        {"imglen": 4, "reflen": 0, "poslen": 5, "isolate": False, "capture": False, "cached": True},
    ]


def test_sample_cfg_reuses_conditional_reference_cache_on_first_step():
    model = _FakeCachedEditModel()
    vae = _FakeVAE()
    txt = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    txtmask = torch.ones(1, 1, dtype=torch.bool)
    untxt = torch.ones_like(txt)
    untxtmask = torch.ones_like(txtmask)

    sample(
        model,
        vae,
        txt,
        txtmask,
        untxt=untxt,
        untxtmask=untxtmask,
        device="cpu",
        dtype=torch.bfloat16,
        width=2,
        height=2,
        steps=1,
        cfg_scale=2.0,
        ref_latents=[torch.ones(1, 2, 2, dtype=torch.bfloat16)],
        kv_cache=True,
    )

    assert model.calls == [
        {"imglen": 8, "reflen": 4, "poslen": 9, "isolate": True, "capture": True, "cached": False},
        {"imglen": 4, "reflen": 0, "poslen": 5, "isolate": False, "capture": False, "cached": True},
    ]
