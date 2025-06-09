import argparse

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from musubi_tuner.utils import model_utils

from blissful_tuner.blissful_logger import BlissfulLogger


logger = BlissfulLogger(__name__, "green")
 


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

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd


def convert_to_diffusers(prefix, weights_sd):
    # convert from default LoRA to diffusers

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

            module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
            module_name = module_name.replace("_", ".")  # replace "_" with "."
            if ".cross.attn." in module_name or ".self.attn." in module_name:
                # Wan2.1 lora name to module name: ugly but works
                module_name = module_name.replace("cross.attn", "cross_attn")  # fix cross attn
                module_name = module_name.replace("self.attn", "self_attn")  # fix self attn
                module_name = module_name.replace("k.img", "k_img")  # fix k img
                module_name = module_name.replace("v.img", "v_img")  # fix v img
            else:
                # HunyuanVideo lora name to module name: ugly but works
                module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
                module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
                module_name = module_name.replace("img.", "img_")  # fix img
                module_name = module_name.replace("txt.", "txt_")  # fix txt
                module_name = module_name.replace("attn.", "attn_")  # fix attn

            diffusers_prefix = "diffusion_model"
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


def convert(input_file, output_file, target_format):
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
        new_weights_sd = convert_to_diffusers(prefix, weights_sd)
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert(args.input, args.output, args.target)


if __name__ == "__main__":
    main()
