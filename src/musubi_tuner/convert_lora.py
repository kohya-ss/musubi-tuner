import argparse

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from musubi_tuner.utils import model_utils

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# keys of Qwen-Image state dict
QWEN_IMAGE_KEYS = [
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "txt_norm",
    "img_in",
    "txt_in",
    "transformer_blocks.*.img_mod.1",
    "transformer_blocks.*.attn.norm_q",
    "transformer_blocks.*.attn.norm_k",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.to_add_out",
    "transformer_blocks.*.attn.norm_added_q",
    "transformer_blocks.*.attn.norm_added_k",
    "transformer_blocks.*.img_mlp.net.0.proj",
    "transformer_blocks.*.img_mlp.net.2",
    "transformer_blocks.*.txt_mod.1",
    "transformer_blocks.*.txt_mlp.net.0.proj",
    "transformer_blocks.*.txt_mlp.net.2",
    "norm_out.linear",
    "proj_out",
]


def convert_from_diffusers(prefix, weights_sd):
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in weights_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            logger.warning(f"unexpected key: {key} in diffusers format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_")
        new_key = new_key.replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")

        # support unknown format: do not replace dots but uses lora_up/lora_up/alpha
        new_key = new_key.replace("_lora_down_", ".lora_down.").replace("_lora_up_", ".lora_up.")
        if new_key.endswith("_alpha"):
            new_key = new_key.replace("_alpha", ".alpha")

        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in new_weights_sd:
            new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def convert_to_diffusers(prefix, diffusers_prefix, weights_sd):
    # convert from default LoRA to diffusers
    if diffusers_prefix is None:
        diffusers_prefix = "diffusion_model"

    # make reverse map from LoRA name to base model module name
    lora_name_to_module_name = {}
    for key in QWEN_IMAGE_KEYS:
        if "*" not in key:
            lora_name = prefix + key.replace(".", "_")
            lora_name_to_module_name[lora_name] = key
        else:
            lora_name = prefix + key.replace(".", "_")
            for i in range(100):  # assume at most 100 transformer blocks
                lora_name_to_module_name[lora_name.replace("*", str(i))] = key.replace("*", str(i))

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue

            lora_name = key.split(".", 1)[0]  # before first dot

            if lora_name in lora_name_to_module_name:
                module_name = lora_name_to_module_name[lora_name]
            else:
                module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
                module_name = module_name.replace("_", ".")  # replace "_" with "."
                if ".cross.attn." in module_name or ".self.attn." in module_name:
                    # Wan2.1 lora name to module name: ugly but works
                    module_name = module_name.replace("cross.attn", "cross_attn")  # fix cross attn
                    module_name = module_name.replace("self.attn", "self_attn")  # fix self attn
                    module_name = module_name.replace("k.img", "k_img")  # fix k img
                    module_name = module_name.replace("v.img", "v_img")  # fix v img
                elif ".attention.to." in module_name or ".feed.forward." in module_name:
                    # Z-Image lora name to module name: ugly but works
                    module_name = module_name.replace("to.q", "to_q")  # fix to q
                    module_name = module_name.replace("to.k", "to_k")  # fix to k
                    module_name = module_name.replace("to.v", "to_v")  # fix to v
                    module_name = module_name.replace("to.out", "to_out")  # fix to out
                    module_name = module_name.replace("feed.forward", "feed_forward")  # fix feed forward
                elif "double.blocks." in module_name or "single.blocks." in module_name:
                    # HunyuanVideo and FLUX lora name to module name: ugly but works
                    module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                    module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                    module_name = module_name.replace("img.", "img_")  # fix img
                    module_name = module_name.replace("txt.", "txt_")  # fix txt
                    module_name = module_name.replace("attn.", "attn_")  # fix attn

            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.warning(f"unexpected key: {key} in default LoRA format")
                continue

            # scale weight by alpha
            if lora_name in lora_alphas:
                # we scale both down and up, so scale is sqrt
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.warning(f"missing alpha for {lora_name}")

            new_weights_sd[new_key] = weight

    return new_weights_sd


def convert(input_file, output_file, target_format, diffusers_prefix):
    logger.info(f"loading {input_file}")
    weights_sd = load_file(input_file)
    with safe_open(input_file, framework="pt") as f:
        metadata = f.metadata()

    logger.info(f"converting to {target_format}")
    prefix = "lora_unet_"
    if target_format == "default":
        new_weights_sd = convert_from_diffusers(prefix, weights_sd)
        metadata = metadata or {}
        model_utils.precalculate_safetensors_hashes(new_weights_sd, metadata)
    elif target_format == "other":
        new_weights_sd = convert_to_diffusers(prefix, diffusers_prefix, weights_sd)
    else:
        raise ValueError(f"unknown target format: {target_format}")

    logger.info(f"saving to {output_file}")
    save_file(new_weights_sd, output_file, metadata=metadata)

    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA weights between default and other formats")
    parser.add_argument("--input", type=str, required=True, help="input model file")
    parser.add_argument("--output", type=str, required=True, help="output model file")
    parser.add_argument("--target", type=str, required=True, choices=["other", "default"], help="target format")
    parser.add_argument(
        "--diffusers_prefix", type=str, default=None, help="prefix for Diffusers weights, default is None (use `diffusion_model`)"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert(args.input, args.output, args.target, args.diffusers_prefix)


if __name__ == "__main__":
    main()
