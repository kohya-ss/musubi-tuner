"""Streaming detail diagnostics in proxy RGB; these are not perceptual quality scores."""

from __future__ import annotations

import math

import torch

from musubi_tuner.dlssnr.losses import masked_gaussian_lowpass
from musubi_tuner.dlssnr.numerics import fp32_execution

HIGHPASS_SIGMAS = (1.0, 4.0)
RATIO_ENERGY_FLOOR = 1e-12
NOISE_SEED_XOR = 0x9E3779B9


def diagnostic_protocol():
    return {
        "schema": "dlssnr_detail_diagnostics_v1",
        "highpass_sigmas_px": list(HIGHPASS_SIGMAS),
        "highpass_operator": "mask_normalized_gaussian_radius_ceil_3sigma_replicate",
        "energy_reduction": "valid_rgb_mean_square",
        "filter_precision": "float32_environment_tf32",
        "sum_dtype": "float64",
        "ratio_energy_floor": RATIO_ENERGY_FLOOR,
        "noise_seed_xor": NOISE_SEED_XOR,
        "noise_pairs_per_frame": 1,
        "noise_history": "shared_primary_history_current_frame_only",
    }


def _masked_sum(value, mask):
    return float((torch.where(mask > 0, value, 0) * mask).sum(dtype=torch.float64))


def _highpass_energy_sum(image, mask, sigma):
    clean = torch.where(mask > 0, image.float(), 0)
    residual = clean - masked_gaussian_lowpass(clean, mask, sigma)
    return _masked_sum(residual.square(), mask)


class DetailDiagnostics:
    """Keep scalar sums only; frame sizes and soft mask coverage set the weights."""

    def __init__(self):
        self.count = 0.0
        self.noise = {name: 0.0 for name in ("rgb_mae", "preclamp_mae")}
        self.energy = {sigma: {name: 0.0 for name in ("input", "target", "output", "noise")} for sigma in HIGHPASS_SIGMAS}

    @torch.no_grad()
    def add(self, output, alternate, tensors):
        with fp32_execution(), torch.autocast(tensors["source"].device.type, enabled=False):
            mask = tensors["loss_mask"].float()
            self.count += 3 * float(mask.sum(dtype=torch.float64))
            delta = output["rendered_proxy"].float() - alternate["rendered_proxy"].float()
            pre_delta = output["neural_preclamp"].float() - alternate["neural_preclamp"].float()
            self.noise["rgb_mae"] += _masked_sum(delta.abs(), mask)
            self.noise["preclamp_mae"] += _masked_sum(pre_delta.abs(), mask)
            images = {"input": tensors["source"], "target": tensors["target"], "output": output["rendered_proxy"], "noise": delta}
            for sigma, sums in self.energy.items():
                for name, image in images.items():
                    sums[name] += _highpass_energy_sum(image, mask, sigma)

    def metrics(self):
        if not all(
            math.isfinite(value)
            for value in (self.count, *self.noise.values(), *(v for s in self.energy.values() for v in s.values()))
        ):
            raise RuntimeError("non-finite detail diagnostics")
        if self.count <= 0:
            raise ValueError("detail diagnostics have no supervised pixels")
        high_frequency, noise_rms = {}, {}
        for sigma, sums in self.energy.items():
            band = f"sigma_{sigma:g}px"
            values = {name: value / self.count for name, value in sums.items()}
            report = {f"{name}_energy": values[name] for name in ("input", "target", "output")}
            for numerator, denominator in (("output", "input"), ("target", "input"), ("output", "target")):
                report[f"{numerator}_to_{denominator}"] = (
                    values[numerator] / values[denominator] if values[denominator] > RATIO_ENERGY_FLOOR else None
                )
            high_frequency[band] = report
            noise_rms[band] = math.sqrt(values["noise"])
        return {
            "protocol": diagnostic_protocol(),
            "valid_rgb_values": self.count,
            "high_frequency": high_frequency,
            "noise_sensitivity": {
                **{name: value / self.count for name, value in self.noise.items()},
                "high_frequency_rms": noise_rms,
            },
        }
