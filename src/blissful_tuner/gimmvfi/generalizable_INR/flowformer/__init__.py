import torch
from .configs.submission import get_cfg
from .core.FlowFormer import build_flowformer


def initialize_Flowformer():
    cfg = get_cfg()
    model = build_flowformer(cfg)

    ckpt = torch.load(cfg.model, map_location="cpu")

    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module" in k}

    ckpt = convert(ckpt)
    model.load_state_dict(ckpt)

    return model
