## Blissful Tuner

[English](./README.md) | [日本語](./README.ja.md)

Blissful extension of Musubi Tuner by Blyss Sarania

Here you will find an extended version of Musubi Tuner with advanced and experimental features focused on creating a full suite of tools for working with generative video models. Preview videos as they generate, increase inference speed, make longer videos and gain more control over your creations and enhance them with VFI, upscaling and more! If you wanna get even more out of Musubi then you've come to the right place! Note for best performance and compatibility, Python 3.12 with PyTorch 2.7.0 or later is recommended! Additional requirements have been added to 'requirements.txt' so if you're coming from regular Musubi, you'll need to `pip install -r requirements.txt` again! While development is done in Python 3.12, efforts are made to maintain compatibility back to 3.10 as well.

Super epic thanks to kohya-ss for his tireless work on Musubi Tuner, kijai for HunyuanVideoWrapper and WanVideoWrapper from which significant code is ported, and all other devs in the open source generative AI community! Please note that due to the experimental nature of many changes, some things might not work as well as the unmodified Musubi! If you find any issues please let me know and I'll do my best to fix them. Please do not post about issues with this version on the main Musubi Github repo but rather use this repo's issues section!

Extensions for all models:
- Latent preview during generation with either latent2RGB or TAEHV (`--preview_latent_every N` where N is a number of steps(or sections for framepack). By default uses latent2rgb, TAE can be enabled with `--preview_vae /path/to/model` models: https://www.dropbox.com/scl/fi/fxkluga9uxu5x6xa94vky/taehv.7z?rlkey=ux1vmcg1yk78gv7iy4iqznpn7&st=4181tzkp&dl=0)
- Optimized generation settings for fast, high quality gens (`--optimized`, enables various optimizations and settings based on the model. Requires SageAttention, Triton, PyTorch 2.7.0 or higher)
- Save generation metadata in videos/images (automatic with `--container mkv`, disable with `--no-metadata`, not available with `--container mp4`)
- Beautiful rich logging, rich argparse and rich tracebacks
- Extended saving options (`--codec codec --container container`, can save Apple ProRes(`--codec prores`, super high bitrate perceptually lossless) into `--container mkv`, or either of `h264`, `h265` into `mp4` or `mkv`)
- FP16 accumulation (`--fp16_accumulation`, works best with Wan FP16 models(but works with Hunyaun bf16 too!) and requires PyTorch 2.7.0 or higher but significantly accelerates inference speeds, especially with `--compile` it's almost as fast as fp8_fast/mmscaled without the loss of precision! And it works with fp8 scaled mode too!)
- GIMM-VFI framerate interpolation (`blissful_tuner/GIMMVFI.py`, please see it's `--help` for usage. Models: https://www.dropbox.com/scl/fi/tcq68jxr52o2gi47eup37/gimm-vfi.7z?rlkey=skvzwxi9lv9455py5wrxv6r5j&st=gu5einkd&dl=0 )
- Upscaling with SwinIR or ESRGAN type models (`blissful_tuner/upscaler.py`, please see it's `--help` for usage. Models: https://www.dropbox.com/scl/fi/wh5hw55o8rofg5mal9uek/upscale.7z?rlkey=oom3osa1zo0pf55092xcfnjp1&st=dozwpzwk&dl=0 )|
- Use strings as your seed because why not! Also easier to remember!
- Use wildcards in your prompts for more variation! (`--prompt_wildcards /path/to/wildcard/directory`, for instance `__color__` in your prompt would look for color.txt in that directory. The wildcard file format is one potential replacement string per line, with an optional relative weight attached like red:2.0 or "some longer string:0.5"  - wildcards can also contain wildcards themselves, the recursion limit is 50 steps!)


Wan/Hunyuan extensions:
- Load diffusion-pipe style LoRAs for inference without converting first
- RifleX e.g. https://github.com/thu-ml/RIFLEx for longer vids (`--riflex_index N` where N is the RifleX frequency. 6 is good for Wan, can usually go to ~115 frames instead of just 81, requires `--rope_func comfy` with Wan; 4 is good for Hunyuan and you can make at least double length!)
- CFGZero* e.g. https://github.com/WeichenFan/CFG-Zero-star (`--cfgzerostar_scaling --cfgzerostar_init_steps N` where N is the total number of steps to 0 out at the start. 2 is good for T2V, 1 for I2V but it's better for T2V in my experience. Support for Hunyuan is HIGHLY experimental and only available with CFG enabled.)
- Advanced CFG scheduling: (`--cfg_schedule`, please see the `--help` for usage. Can specify guidance scale down to individual steps if you like!)
- Perpendicular Negative Guidance (`--perp_neg neg_strength`, where neg_strength is a float that controls the string of the negative prompt. See `--help` for more!)

Hunyuan only extensions:
- Several more LLM options (`--hidden_state_skip_layer N --apply_final_norm --reproduce`, please see the `--help` for explanations!)
- FP8 scaled support using the same algo as Wan (`--fp8_scaled`, Training isn't super tested!)
- Separate prompt for CLIP (`--prompt_2 "second prompt goes here"`, provides a different prompt to CLIP since it's used to simpler text)
- Rescale text encoders based on https://github.com/zer0int/ComfyUI-HunyuanVideo-Nyan (`--te_multiplier llm clip` such as `--te_multiplier 0.9 1.2` to downweight the LLM slightly and upweight the CLIP slightly)

Wan only extensions(now supporting both one shot and interactive modes):
- V2V inferencing (`--video_path /path/to/input/video --v2v_denoise amount` where amount is a float 0.0 - 1.0 that controls how strong the noise added to the source video will be. If `--v2v_noise_mode traditional` then it will run the last (amount * 100) percent of the timestep schedule like other implementations. If `--v2v_noise_mode direct` it will directly control the amount of noise added as closely as possible by starting from wherever in the timestep schedule is closest to that value and proceeding from there. Supports scaling, padding, and truncation so the input doesn't have to be the same res as the output or even the same length! If `--video_length` is shorter than the input, the input will be truncated and include only the first `--video_length` frames. If `--video_length` is longer than the input, the first frame or last frame will be repeated to pad the length depending on `--v2v_pad_mode`. You can use either T2V or I2V `--task` modes and models(i2v mode produces better quality in my opinion)! In I2V mode, if `--image_path` is not specified, the first frame of the video will be used to condition the model instead. `--infer_steps` should be the same amount it would for a full denoise e.g. by default 50 for T2V or 40 for I2V because we need to modify from a full schedule. Actual steps will depend on `--v2v_noise_mode`)
- Prompt weighting (`--prompt_weighting` and then in your prompt you can do like "a cat playing with a (large:1.4) red ball" to upweight the effect of "large". Note that [this] or (this) isn't supported, only (this:1.0) and also downweighting has curious effects
- ROPE ported from ComfyUI that doesn't use complex numbers. Massive VRAM savings when used with `--compile`! (`--rope_func comfy`)
- Optional extra latent noise for I2V/V2V (`--v2v_extra_noise 0.02 --i2v_extra_noise 0.02`, values less than 0.04 are recommended. This can improve fine detail and texture in V2V/I2V but too much will cause artifacts and moving shadows. I use around 0.01-0.02 for V2V and 0.02-0.04 for I2V)

Framepack only extensions:
- Torch.compile (`--compile`, same syntax as Wan and Hunyuan already use)
- FP8 fast/mm_scaled (`--fp8_fast`, increased speed on 40xx cards with a mild hit to quality, Wan and Hunyuan have this already in native Musubi!)

My general code and Musubi Tuner code is licensed Apache 2.0. Other projects included may have different licensing, in which case you will find a LICENSE file in their directory specifying the terms under which they are included! Below is the original Musubi Readme which still remains relevant:

# Musubi Tuner Readme

## Table of Contents

- [Musubi Tuner](#musubi-tuner)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Support the Project](#support-the-project)
    - [Recent Updates](#recent-updates)
    - [Releases](#releases)
  - [Overview](#overview)
    - [Hardware Requirements](#hardware-requirements)
    - [Features](#features)
  - [Installation](#installation)
    - [pip based installation](#pip-based-installation)
    - [uv based installation](#uv-based-installation)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
  - [Model Download](#model-download)
    - [Use the Official HunyuanVideo Model](#use-the-official-hunyuanvideo-model)
    - [Using ComfyUI Models for Text Encoder](#using-comfyui-models-for-text-encoder)
  - [Usage](#usage)
    - [Dataset Configuration](#dataset-configuration)
    - [Latent Pre-caching](#latent-pre-caching)
    - [Text Encoder Output Pre-caching](#text-encoder-output-pre-caching)
    - [Configuration of Accelerate](#configuration-of-accelerate)
    - [Training](#training)
    - [Merging LoRA Weights](#merging-lora-weights)
    - [Inference](#inference)
    - [Inference with SkyReels V1](#inference-with-skyreels-v1)
    - [Convert LoRA to another format](#convert-lora-to-another-format)
  - [Miscellaneous](#miscellaneous)
    - [SageAttention Installation](#sageattention-installation)
    - [PyTorch version](#pytorch-version)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo, Wan2.1 and FramePack architectures. 

This repository is unofficial and not affiliated with the official HunyanVideo/Wan2.1/FramePack repositories. 

For Wan2.1, please also refer to [Wan2.1 documentation](./docs/wan.md). For FramePack, please also refer to [FramePack documentation](./docs/framepack.md).

*This repository is under development.*

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Recent Updates

- GitHub Discussions Enabled: We've enabled GitHub Discussions for community Q&A, knowledge sharing, and technical information exchange. Please use Issues for bug reports and feature requests, and Discussions for questions and sharing experiences. [Join the conversation →](https://github.com/kohya-ss/musubi-tuner/discussions)

- May 11, 2025:
    - Added support for single frame training in FramePack. This is an experimental feature that allows training for single frame inference. See [FramePack documentation](./docs/framepack.md#single-frame-training--1フレーム学習) for details.

- May 9, 2025 update 2:
    - Added support for single frame inference in FramePack. This is a unique feature of this repository that generates an image after time progression according to the prompt, rather than a video. In other words, it allows limited natural language editing of images. See [FramePack documentation](./docs/framepack.md#single-frame-inference--単一フレーム推論) for details.
    - Added `--video_sections` option to specify the length of the generated video in terms of sections instead of seconds in the inference code of FramePack. Also added `--output_type latent_images` (saves both latent and images).

- May 9, 2025:
    - Added support for applying LoRA for HunyuanVideo in the inference code of FramePack. Both LoRA from this repository and diffusion-pipe can be applied. See [FramePack documentation](./docs/framepack.md#inference) for details.

- May 4, 2025:
    - Added training and inference for FramePack-F1 (experimental feature). See [FramePack documentation](./docs/framepack.md) for details. 
        - Please re-cache the latents for FramePack-F1 with `--f1` option (`--vanilla_sampling` is changed to `--f1`, and the behavior is changed). The cache files are not compatible with FramePack. The cache files cannot be shared between FramePack and FramePack-F1, so please specify a different `.toml` file for another cache directory. 

- May 1, 2025:
    - Added features to the inference code of FramePack, such as latent padding specification and custom prompt specification. See [FramePack documentation](./docs/framepack.md#inference) for details.
        - The behavior when specifying the section start image has changed (it no longer automatically sets latent padding to 0, so the start image is used as a reference image). To maintain the previous behavior (force the section start image), specify `--latent_padding 0,0,0,0` (specify 0 for each section).
        
- Apr 26, 2025:
    - Added inference and LoRA training for FramePack. PR [#230](https://github.com/kohya-ss/musubi-tuner/pull/230). See [FramePack documentation](./docs/framepack.md) for details.

- Apr 18, 2025:
    - Added batch generation mode that reads prompts from a file and generates them during Wan2.1 inference, as well as interactive mode that specifies prompts from the command line. See [here](./docs/wan.md#interactive-mode--インタラクティブモード) for details.


### Releases

We are grateful to everyone who has been contributing to the Musubi Tuner ecosystem through documentation and third-party tools. To support these valuable contributions, we recommend working with our [releases](https://github.com/kohya-ss/musubi-tuner/releases) as stable reference points, as this project is under active development and breaking changes may occur.

You can find the latest release and version history in our [releases page](https://github.com/kohya-ss/musubi-tuner/releases).

## Overview

### Hardware Requirements

- VRAM: 12GB or more recommended for image training, 24GB or more for video training
    - *Actual requirements depend on resolution and training settings.* For 12GB, use a resolution of 960x544 or lower and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`, etc.
- Main Memory: 64GB or more recommended, 32GB + swap may work

### Features

- Memory-efficient implementation
- Windows compatibility confirmed (Linux compatibility confirmed by community)
- Multi-GPU support not implemented

## Installation

### pip based installation

Python 3.10 or later is required (verified with 3.10).

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. 

PyTorch 2.5.1 or later is required (see [note](#PyTorch-version)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

Optionally, you can use FlashAttention and SageAttention (**for inference only**; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Optional dependencies for additional features:
- `ascii-magic`: Used for dataset verification
- `matplotlib`: Used for timestep visualization
- `tensorboard`: Used for logging training progress

```bash
pip install ascii-magic matplotlib tensorboard
```

### uv based installation (experimenal)

You can also install using uv, but installation with uv is experimental. Feedback is welcome.

1. Install uv (if not already present on your OS).

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Follow the instructions to add the uv path manually until you restart your session...

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Follow the instructions to add the uv path manually until you reboot your system... or just reboot your system at this point.

## Model Download

There are two ways to download the model.

### Use the Official HunyuanVideo Model

Download the model following the [official README](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md) and place it in your chosen directory with the following structure:

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Using ComfyUI Models for Text Encoder

This method is easier.

For DiT and VAE, use the HunyuanVideo models.

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers, download [mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) and place it in your chosen directory.

(Note: The fp8 model on the same page is unverified.)

If you are training with `--fp8_base`, you can use `mp_rank_00_model_states_fp8.safetensors` from [here](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial) instead of `mp_rank_00_model_states.pt`. (This file is unofficial and simply converts the weights to float8_e4m3fn.)

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae, download [pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) and place it in your chosen directory.

For the Text Encoder, use the models provided by ComfyUI. Refer to [ComfyUI's page](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/), from https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders, download `llava_llama3_fp16.safetensors` (Text Encoder 1, LLM) and `clip_l.safetensors` (Text Encoder 2, CLIP)  and place them in your chosen directory.

(Note: The fp8 LLM model on the same page is unverified.)

## Usage

### Dataset Configuration

Please refer to [dataset configuration guide](./dataset/dataset_config.md).

### Latent Pre-caching

Latent pre-caching is required. Create the cache using the following command:

If you have installed using pip:

```bash
python cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

If you have installed with `uv`, you can use `uv run` to run the script. Other scripts can be run in the same way. (Note that the installation with `uv` is experimental. Feedback is welcome. If you encounter any issues, please use the pip-based installation.)

```bash
uv run cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

For additional options, use `python cache_latents.py --help`.

If you're running low on VRAM, reduce `--vae_spatial_tile_sample_min_size` to around 128 and lower the `--batch_size`.

Use `--debug_mode image` to display dataset images and captions in a new window, or `--debug_mode console` to display them in the console (requires `ascii-magic`). 

With `--debug_mode video`, images or videos will be saved in the cache directory (please delete them after checking). The bitrate of the saved video is set to 1Mbps for preview purposes. The images decoded from the original video (not degraded) are used for the cache (for training).

When `--debug_mode` is specified, the actual caching process is not performed.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

### Text Encoder Output Pre-caching

Text Encoder output pre-caching is required. Create the cache using the following command:

```bash
python cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

or for uv:

```bash
uv run cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

For additional options, use `python cache_text_encoder_outputs.py --help`.

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_llm` to run the LLM in fp8 mode.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

### Configuration of Accelerate

Run `accelerate config` to configure Accelerate. Choose appropriate values for each question based on your environment (either input values directly or use arrow keys and enter to select; uppercase is default, so if the default value is fine, just press enter without inputting anything). For training with a single GPU, answer the questions as follows:


```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

*Note*: In some cases, you may encounter the error `ValueError: fp16 mixed precision requires a GPU`. If this happens, answer "0" to the sixth question (`What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`). This means that only the first GPU (id `0`) will be used.

### Training

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

or for uv:

```bash
uv run accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

__Update__: Changed the sample training settings to a learning rate of 2e-4, `--timestep_sampling` to `shift`, and `--discrete_flow_shift` to 7.0. Faster training is expected. If the details of the image are not learned well, try lowering the discete flow shift to around 3.0.

However, the training settings are still experimental. Appropriate learning rates, training steps, timestep distribution, loss weighting, etc. are not yet known. Feedback is welcome.

For additional options, use `python hv_train_network.py --help` (note that many options are unverified).

Specifying `--fp8_base` runs DiT in fp8 mode. Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality. If `--fp8_base` is not specified, 24GB or more VRAM is recommended. Use `--blocks_to_swap` as needed.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 36.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

Use `--sdpa` for PyTorch's scaled dot product attention. Use `--flash_attn` for [FlashAttention](https://github.com/Dao-AILab/flash-attention). Use `--xformers` for xformers, but specify `--split_attn` when using xformers. `--sage_attn` for SageAttention, but SageAttention is not yet supported for training, so it raises a ValueError.

`--split_attn` processes attention in chunks. Speed may be slightly reduced, but VRAM usage is slightly reduced.

The format of LoRA trained is the same as `sd-scripts`.

`--show_timesteps` can be set to `image` (requires `matplotlib`) or `console` to display timestep distribution and loss weighting during training.

You can record logs during training. Refer to [Save and view logs in TensorBoard format](./docs/advanced_config.md#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照).

For PyTorch Dynamo optimization, refer to [this document](./docs/advanced_config.md#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化).

For sample image generation during training, refer to [this document](./docs/sampling_during_training.md). For advanced configuration, refer to [this document](./docs/advanced_config.md).

### Merging LoRA Weights

Note: Wan2.1 is not supported for merging LoRA weights.

```bash
python merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

or for uv:

```bash
uv run merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

Specify the device to perform the calculation (`cpu` or `cuda`, etc.) with `--device`. Calculation will be faster if `cuda` is specified.

Specify the LoRA weights to merge with `--lora_weight` and the multiplier for the LoRA weights with `--lora_multiplier`. Multiple values can be specified, and the number of values must match.

### Inference

Generate videos using the following command:

```bash
python hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

or for uv:

```bash
uv run hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

For additional options, use `python hv_generate_video.py --help`.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

`--fp8_fast` option is also available for faster inference on RTX 40x0 GPUs. This option requires `--fp8` option. 

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 38.

For `--attn_mode`, specify either `flash`, `torch`, `sageattn`, `xformers`, or `sdpa` (same as `torch`). These correspond to FlashAttention, scaled dot product attention, SageAttention, and xformers, respectively. Default is `torch`. SageAttention is effective for VRAM reduction.

Specifing `--split_attn` will process attention in chunks. Inference with SageAttention is expected to be about 10% faster.

For `--output_type`, specify either `both`, `latent`, `video` or `images`. `both` outputs both latents and video. Recommended to use `both` in case of Out of Memory errors during VAE processing. You can specify saved latents with `--latent_path` and use `--output_type video` (or `images`) to only perform VAE decoding.

`--seed` is optional. A random seed will be used if not specified.

`--video_length` should be specified as "a multiple of 4 plus 1".

`--flow_shift` can be specified to shift the timestep (discrete flow shift). The default value when omitted is 7.0, which is the recommended value for 50 inference steps. In the HunyuanVideo paper, 7.0 is recommended for 50 steps, and 17.0 is recommended for less than 20 steps (e.g. 10).

By specifying `--video_path`, video2video inference is possible. Specify a video file or a directory containing multiple image files (the image files are sorted by file name and used as frames). An error will occur if the video is shorter than `--video_length`. You can specify the strength with `--strength`. It can be specified from 0 to 1.0, and the larger the value, the greater the change from the original video.

Note that video2video inference is experimental.

`--compile` option enables PyTorch's compile feature (experimental). Requires triton. On Windows, also requires Visual C++ build tools installed and PyTorch>=2.6.0 (Visual C++ build tools is also required). You can pass arguments to the compiler with `--compile_args`.

The `--compile` option takes a long time to run the first time, but speeds up on subsequent runs.

You can save the DiT model after LoRA merge with the `--save_merged_model` option. Specify `--save_merged_model path/to/merged_model.safetensors`. Note that inference will not be performed when this option is specified.

### Inference with SkyReels V1

SkyReels V1 T2V and I2V models are supported (inference only). 

The model can be downloaded from [here](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy). Many thanks to Kijai for providing the model. `skyreels_hunyuan_i2v_bf16.safetensors` is the I2V model, and `skyreels_hunyuan_t2v_bf16.safetensors` is the T2V model. The models other than bf16 are not tested (`fp8_e4m3fn` may work).

For T2V inference, add the following options to the inference command:

```bash
--guidance_scale 6.0 --embedded_cfg_scale 1.0 --negative_prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" --split_uncond
```

SkyReels V1 seems to require a classfier free guidance (negative prompt).`--guidance_scale` is a guidance scale for the negative prompt. The recommended value is 6.0 from the official repository. The default is 1.0, it means no classifier free guidance.

`--embedded_cfg_scale` is a scale of the embedded guidance. The recommended value is 1.0 from the official repository (it may mean no embedded guidance).

`--negative_prompt` is a negative prompt for the classifier free guidance. The above sample is from the official repository. If you don't specify this, and specify `--guidance_scale` other than 1.0, an empty string will be used as the negative prompt.

`--split_uncond` is a flag to split the model call into unconditional and conditional parts. This reduces VRAM usage but may slow down inference. If `--split_attn` is specified, `--split_uncond` is automatically set.

You can also perform image2video inference with SkyReels V1 I2V model. Specify the image file path with `--image_path`. The image will be resized to the given `--video_size`.

```bash
--image_path path/to/image.jpg
``` 

### Convert LoRA to another format

You can convert LoRA to a format compatible with ComfyUI (presumed to be Diffusion-pipe) using the following command:

```bash
python convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

or for uv:

```bash
uv run convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

Specify the input and output file paths with `--input` and `--output`, respectively.

Specify `other` for `--target`. Use `default` to convert from another format to the format of this repository.

Wan2.1 is also supported. 

## Miscellaneous

### SageAttention Installation

sdbsd has provided a Windows-compatible SageAttention implementation and pre-built wheels here:  https://github.com/sdbds/SageAttention-for-windows. After installing triton, if your Python, PyTorch, and CUDA versions match, you can download and install the pre-built wheel from the [Releases](https://github.com/sdbds/SageAttention-for-windows/releases) page. Thanks to sdbsd for this contribution.

For reference, the build and installation instructions are as follows. You may need to update Microsoft Visual C++ Redistributable to the latest version.

1. Download and install triton 3.1.0 wheel matching your Python version from [here](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Install Microsoft Visual Studio 2022 or Build Tools for Visual Studio 2022, configured for C++ builds.

3. Clone the SageAttention repository in your preferred directory:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. Open `x64 Native Tools Command Prompt for VS 2022` from the Start menu under Visual Studio 2022.

5. Activate your venv, navigate to the SageAttention folder, and run the following command. If you get a DISTUTILS not configured error, set `set DISTUTILS_USE_SDK=1` and try again:
    ```shell
    python setup.py install
    ```

This completes the SageAttention installation.

### PyTorch version

If you specify `torch` for `--attn_mode`, use PyTorch 2.5.1 or later (earlier versions may result in black videos).

If you use an earlier version, use xformers or SageAttention.

## Disclaimer

This repository is unofficial and not affiliated with the official HunyuanVideo repository. 

This repository is experimental and under active development. While we welcome community usage and feedback, please note:

- This is not intended for production use
- Features and APIs may change without notice
- Some functionalities are still experimental and may not work as expected
- Video training features are still under development

If you encounter any issues or bugs, please create an Issue in this repository with:
- A detailed description of the problem
- Steps to reproduce
- Your environment details (OS, GPU, VRAM, Python version, etc.)
- Any relevant error messages or logs

## Contributing

We welcome contributions! However, please note:

- Due to limited maintainer resources, PR reviews and merges may take some time
- Before starting work on major changes, please open an Issue for discussion
- For PRs:
  - Keep changes focused and reasonably sized
  - Include clear descriptions
  - Follow the existing code style
  - Ensure documentation is updated

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Code under the `wan` directory is modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The license is under the Apache License 2.0.

Code under the `frame_pack` directory is modified from [FramePack](https://github.com/lllyasviel/FramePack). The license is under the Apache License 2.0.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
