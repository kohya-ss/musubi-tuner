# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import json
import logging
import struct
from math import floor, sqrt
from typing import Union, Optional

import transformers
import torch
import torchvision
import torchvision.transforms.functional as F
from peft import PeftConfig, LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import ToPILImage

from .generation_utils import generate_sample_i2v

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


def resize_image(image, max_area, divisibility=16):
    h, w = image.shape[2:]
    area = h * w
    k = sqrt(max_area / area) / divisibility
    new_h = int(round(h * k) * divisibility)
    new_w = int(round(w * k) * divisibility)
    return F.resize(image, (new_h, new_w)), k


def get_first_frame_from_image(image, vae, device, max_area, divisibility):
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"unknown image type: {type(image)}")

    image = F.pil_to_tensor(pil_image).unsqueeze(0)
    image, k = resize_image(image, max_area=max_area, divisibility=divisibility)
    image = image / 127.5 - 1.0

    with torch.no_grad():
        image = image.to(device=device, dtype=torch.float16).transpose(0, 1).unsqueeze(0)
        lat_image = vae.encode(image, opt_tiling=False).latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)
        lat_image = lat_image * vae.config.scaling_factor

    return pil_image, lat_image, k


def read_safetensors_json(file_path):
    """Reads the metadata (JSON header) from a safetensors file."""
    with open(file_path, "rb") as f:
        # Step 1: Read the first 8 bytes to get the size of the header (N)
        header_size_bytes = f.read(8)
        header_size = struct.unpack("Q", header_size_bytes)[0]  # 'Q' is for unsigned 64-bit integer

        # Step 2: Read the next N bytes which contain the JSON header
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode("utf-8")

        # Step 3: Parse the JSON header
        header = json.loads(header_str)
        return header


class Kandinsky5I2VPipeline:
    def __init__(
        self,
        device_map: Union[str, torch.device, dict],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf=None,
        offload: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = conf.metrics.resolution
        self.max_area = 512 * 768 if self.resolution == 512 else 1024 * 1024
        self.divisibility = 16 if self.resolution == 512 else 128

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload

        self._hf_peft_config_loaded = False
        self.peft_config = {}
        self.peft_triggers = {}
        self.peft_trigger = ""
        self.device_mesh = device_mesh

    def __call__(
        self,
        text: str,
        image: Union[str, Image.Image],
        time_length: int = 5,  # time in seconds 0 if you want generate image
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        # PREPARATION
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        if self.offload:
            self.vae = self.vae.to(self.device_map["vae"], non_blocking=True)
        image, image_lat, k = get_first_frame_from_image(image, self.vae, self.device_map["vae"], self.max_area, self.divisibility)
        if self.offload:
            self.vae = self.vae.to("cpu", non_blocking=True)

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.text_embedder.embedder.expand_text_prompt(caption, image, device=self.device_map["text_embedder"])
                caption = self.peft_trigger + caption
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        height, width = image_lat.shape[1:3]
        shape = (1, num_frames, height, width, 16)

        # GENERATION
        images = generate_sample_i2v(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            images=image_lat,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            progress=progress,
            offload=self.offload,
            tp_mesh=self.device_mesh,
        )
        torch.cuda.empty_cache()

        if self.offload:
            self.text_embedder = self.text_embedder.to(device=self.device_map["text_embedder"])

        if k > 16:
            h, w = images.shape[-2:]
            images = F.resize(images[0], (int(h / k / 16), int(w / k / 16)))

        # RESULTS
        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for image in images.squeeze(2).cpu():
                    return_images.append(ToPILImage()(image))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(return_images):
                        for path, image in zip(save_path, return_images):
                            image.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(images):
                        for path, video in zip(save_path, images):
                            torchvision.io.write_video(
                                path,
                                video.float().permute(1, 2, 3, 0).cpu().numpy(),
                                fps=24,
                                options={"crf": "5"},
                            )
                return images

    def load_adapter(
        self,
        adapter_config: Union[PeftConfig, str],
        adapter_path: Optional[str] = None,
        adapter_name: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> None:
        if adapter_name is None:
            adapter_name = "default"
        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            try:
                with open(adapter_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config = LoraConfig(**adapter_config)
            except:
                raise TypeError(f"adapter_config should be an instance of PeftConfig or a path to a json file.")
        self.peft_config[adapter_name] = adapter_config

        inject_adapter_in_model(adapter_config, self.dit, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        adapter_state_dict = load_file(adapter_path)
        adapter_metadata = read_safetensors_json(adapter_path)
        if trigger is not None:
            self.peft_trigger = trigger
        else:
            if "__metadata__" in adapter_metadata and "trigger" in adapter_metadata["__metadata__"]:
                self.peft_trigger = adapter_metadata["__metadata__"]["trigger"]
            else:
                self.peft_trigger = ""
        self.peft_triggers[adapter_name] = self.peft_trigger

        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = key
            for prefix in ["base_model.model.", "transformer."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break

            new_key = new_key.replace(".default", "")
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(self.dit, processed_adapter_state_dict, adapter_name)

        if incompatible_keys is not None:
            err_msg = ""
            origin_name = "state_dict"
            # Check for unexpected keys.
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                err_msg = (
                    f"Loading adapter weights from {origin_name} led to unexpected keys not found in the model: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )

            # Check for missing keys.
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                # Filter missing keys specific to the current adapter, as missing base model keys are expected.
                lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                if lora_missing_keys:
                    err_msg += (
                        f"Loading adapter weights from {origin_name} led to missing keys in the model: "
                        f"{', '.join(lora_missing_keys)}"
                    )

            if err_msg:
                logging.warning(err_msg)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[list[str], str]) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.dit.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatibility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
        self.peft_trigger = self.peft_triggers[adapter_name]

    def disable_adapters(self) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.dit.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True
        self.peft_trigger = ""
