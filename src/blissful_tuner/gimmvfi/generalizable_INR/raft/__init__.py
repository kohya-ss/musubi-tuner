from .raft import RAFT
import argparse
import torch
from .extractor import BasicEncoder


def initialize_RAFT(model_path="pretrained_ckpt/raft-things.pth", device="cuda"):
    """Initializes the RAFT model."""
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = RAFT(args)
    ckpt = torch.load(args.raft_model, map_location="cpu")

    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module" in k}

    ckpt = convert(ckpt)
    model.load_state_dict(ckpt, strict=True)
    print("load raft from " + model_path)

    return model
