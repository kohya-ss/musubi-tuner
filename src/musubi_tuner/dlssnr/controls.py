"""Fixed controls encoded as OpenDLSS-NR feature lanes 10-14.

See the pinned nr_preprocess.comp reference in NOTICE.md. Auto mask is a
network conditioning switch, not a generated supervision mask.
"""

import math

import torch

CONTROL_DEFAULTS = {
    "nr_style": 0,
    "nr_tone": 1.0,
    "nr_structure": 1.0,
    "nr_skin": -1.0,
    "nr_auto_mask": True,
}


def resolve_fixed_controls(values):
    resolved = {name: values.get(name, default) for name, default in CONTROL_DEFAULTS.items()}
    if type(resolved["nr_style"]) is not int or not 0 <= resolved["nr_style"] <= 2**24:
        raise ValueError("nr_style must be a nonnegative exactly representable integer <= 16777216")
    if type(resolved["nr_auto_mask"]) is not bool:
        raise ValueError("nr_auto_mask must be a boolean")
    for name in ("nr_tone", "nr_structure", "nr_skin"):
        value = resolved[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if not 0 <= value <= 1 and not (name == "nr_skin" and value == -1):
            raise ValueError(f"{name} must be in [0,1]" + (" or -1 to follow structure" if name == "nr_skin" else ""))
    return resolved


def fixed_control_tensor(values, width, height):
    values = resolve_fixed_controls(values)
    tone, structure, skin = (values[f"nr_{name}"] for name in ("tone", "structure", "skin"))
    lanes = [tone, 1.0, structure if skin == -1 else skin, structure] if values["nr_auto_mask"] else [tone, structure, -1, -1]
    encoded = torch.tensor([values["nr_style"] / 128, *lanes], dtype=torch.float32)
    encoded[1:] = encoded[1:].half().float()
    return encoded[:, None, None].expand(5, height, width)
