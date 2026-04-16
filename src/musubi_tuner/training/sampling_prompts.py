"""Prompt loading and sampling trigger helpers used during training."""

import json
import logging
import re
from typing import Dict

import toml


logger = logging.getLogger(__name__)


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"f (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["frame_count"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"fs ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["discrete_flow_shift"] = float(m.group(1))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["cfg_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:  # image path
                prompt_dict["image_path"] = m.group(1).strip()
                continue

            m = re.match(r"ei (.+)", parg, re.IGNORECASE)
            if m:  # end image path
                prompt_dict["end_image_path"] = m.group(1).strip()
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["control_video_path"] = m.group(1).strip()
                continue

            m = re.match(r"ci (.+)", parg, re.IGNORECASE)
            if m:
                # can be multiple control images
                control_image_path = m.group(1).strip()
                if "control_image_path" not in prompt_dict:
                    prompt_dict["control_image_path"] = []
                prompt_dict["control_image_path"].append(control_image_path)
                continue

            m = re.match(r"of (.+)", parg, re.IGNORECASE)
            if m:  # output folder
                prompt_dict["one_frame"] = m.group(1).strip()
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def should_sample_images(args, steps, epoch=None):
    if steps == 0:
        if not args.sample_at_first:
            return False
    else:
        should_sample_by_steps = args.sample_every_n_steps is not None and steps % args.sample_every_n_steps == 0
        should_sample_by_epochs = (
            args.sample_every_n_epochs is not None and epoch is not None and epoch % args.sample_every_n_epochs == 0
        )
        if not should_sample_by_steps and not should_sample_by_epochs:
            return False
    return True
