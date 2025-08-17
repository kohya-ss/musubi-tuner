import torch
from torch import nn

# region activations


ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


# endregion activations
