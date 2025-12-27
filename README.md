## Blissful Tuner

[English](./README.md) | [日本語](./README.ja.md)

Blissful extension of Musubi Tuner by Blyss Sarania

Here you will find an extended version of Musubi Tuner with advanced and experimental features focused on creating a full suite of tools for working with generative video models. Preview videos as they generate, increase inference speed, make longer videos and gain more control over your creations and enhance them with VFI, upscaling and more! If you wanna get even more out of Musubi then you've come to the right place! Note for best performance and compatibility, Python 3.12 with PyTorch 2.7.0 or later is recommended! While development is done in Python 3.12, efforts are made to maintain compatibility back to 3.10 as well.

IMPORTANT NOTE: Please only install either regular Musubi Tuner or Blissful Tuner into the same venv and uninstall the existing one (e.g. `pip uninstall blissful-tuner`) when switching between Musubi and Blissful. Blissful Tuner is built directly on top of Musubi Tuner and shares many files with it, switching without this step can cause many issues. Thanks!

Super epic thanks to kohya-ss for his tireless work on Musubi Tuner, kijai for HunyuanVideoWrapper and WanVideoWrapper from which significant code is ported, and all other devs in the open source generative AI community! Please note that due to the experimental nature of many changes, some things might not work as well as the unmodified Musubi! If you find any issues please let me know and I'll do my best to fix them. Please do not post about issues with this version on the main Musubi Github repo but rather use this repo's issues section!

In order to keep this section maintainable as the project grows, each feature will be listed once along with a legend indicating which models in the project currently support that feature. Most features pertain to inference, if a feature is available for training that will be specifically noted. Many smaller optimizations and features too numerous to list have been done as well. For the latest updates, I maintain something of a devlog [here](https://github.com/kohya-ss/musubi-tuner/discussions/232)

Legend of current models: Hunyuan Video: (HY), Wan 2.1/2.2: (WV), Framepack: (FP), Flux (FX), Qwen Image (QI), Available for training: (T)

Blissful Features:
- Beautiful rich logging, rich argparse and rich tracebacks (HY) (WV) (FP) (FX) (QI) (T)
- Use wildcards in your prompts for more variation! (`--prompt_wildcards /path/to/wildcard/directory`, for instance `__color__` in your prompt would look for color.txt in that directory. The wildcard file format is one potential replacement string per line, with an optional relative weight attached like red:2.0 or "some longer string:0.5"  - wildcards can also contain wildcards themselves, the recursion limit is 50 steps!) (HY) (WV) (FP) (FX) (QI)
- Use strings as your seed because why not! Also easier to remember! (HY) (WV) (FP) (FX) (QI)
- Powerful, global seed per generation to ensure determinism (HY) (WV) (FP) (FX) (QI)
- Load foreign LoRAs for inference without converting first (HY) (WV) (FP) (FX) (QI)
- Latent preview during generation with either latent2RGB or TAEHV (`--preview_latent_every N` where N is a number of steps(or sections for framepack). By default uses latent2rgb, TAE can be enabled with `--preview_vae /path/to/model` [models](https://huggingface.co/Blyss/BlissfulModels/tree/main/taehv)) (HY) (WV) (FP) (FX)
- Optimized generation settings for fast, high quality gens* (`--optimized`\*, enables various optimizations and settings based on the model. Requires SageAttention, Triton, PyTorch 2.7.0 or higher) (HY) (WV) (FP) (FX)
- FP16 accumulation (`--fp16_accumulation`, works best with Wan FP16 models(but works with Hunyaun bf16 too!) and requires PyTorch 2.7.0 or higher but significantly accelerates inference speeds, especially with `--compile`\* it's almost as fast as fp8_fast/mmscaled without the loss of precision! And it works with fp8 scaled mode too!) (HY) (WV) (FP) (FX)
- Extended saving options (`--codec codec --container container`, can save Apple ProRes(`--codec prores`, super high bitrate perceptually lossless) into `--container mkv`, or either of `h264`, `h265` into `mp4` or `mkv`)  (HY) (WV) (FP)
- Save generation metadata in videos/images (automatic with `--container mkv` and when saving PNG, disable with `--no-metadata`, not available with `--container mp4` You can conveniently view/copy such metadata with `src/blissful_tuner/metaview.py some_video.mkv`, the viewer requires mediainfo_cli) (HY) (WV) (FP) (FX)
- [CFGZero*](https://github.com/WeichenFan/CFG-Zero-star) (`--cfgzerostar_scaling --cfgzerostar_init_steps N` where N is the total number of steps to 0 out at the start. 2 is good for T2V, 1 for I2V but it's better for T2V in my experience. Support for Hunyuan is HIGHLY experimental and only available with CFG enabled.) (HY) (WV) (FX)
- Advanced CFG scheduling: (`--cfg_schedule`, please see the `--help` for usage. Can specify guidance scale down to individual steps if you like!) (HY) (WV) (FX)
- [RifleX](https://github.com/thu-ml/RIFLEx) for longer vids (`--riflex_index N` where N is the RifleX frequency. 6 is good for Wan, can usually go to ~115 frames instead of just 81, requires `--rope_func comfy` with Wan; 4 is good for Hunyuan and you can make at least double length!) (HY) (WV)
- Perpendicular Negative Guidance (`--perp_neg neg_strength`, where neg_strength is a float that controls the string of the negative prompt. See `--help` for more!) (HY) (WV)
- [Normalized Attention Guidance (NAG)](https://arxiv.org/pdf/2505.21179) (Provides negative guidance within cross attention layers. Works for distilled models as well as with regular CFG! Enable with `--nag_scale 3.0` and provide a negative prompt!) (WV)
- Distilled sampling with high quality and low steps (Use `--sample_solver lcm` or `--sample_solver dpm++sde` with distilled Wan models/LoRA like [lightx2v's](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) or with the base model and [my convenient LoRA](https://huggingface.co/Blyss/BlissfulModels/tree/main/wan_lcm)) (WV)
- V2V inferencing (`--video_path /path/to/input/video --denoise_strength amount` where amount is a float 0.0 - 1.0 that controls how strong the noise added to the source video will be. If `--noise_mode traditional` then it will run the last (amount \* 100) percent of the timestep schedule like other implementations. If `--noise_mode direct` it will directly control the amount of noise added as closely as possible by starting from wherever in the timestep schedule is closest to that value and proceeding from there. Supports scaling, padding, and truncation so the input doesn't have to be the same res as the output or even the same length! If `--video_length` is shorter than the input, the input will be truncated and include only the first `--video_length` frames. If `--video_length` is longer than the input, the first frame or last frame will be repeated to pad the length depending on `--v2v_pad_mode`. You can use either T2V or I2V `--task` modes and models(i2v mode produces better quality in my opinion)! In I2V mode, if `--image_path` is not specified, the first frame of the video will be used to condition the model instead. `--infer_steps` should be the same amount it would for a full denoise e.g. by default 50 for T2V or 40 for I2V because we need to modify from a full schedule. Actual steps will depend on `--noise_mode`) (WV)
- I2I inferencing (`--i2i_path /path/to/image` - use with T2V model in T2I mode, specify strength with `--denoise_strength`. Supports `--i2_extra_noise` for latent noise augmentation as well) (WV)
- Prompt weighting (`--prompt_weighting` and then in your prompt you can do like "a cat playing with a (large:1.4) red ball" to upweight the effect of "large". Note that [this] or (this) isn't supported, only (this:1.0) (WV) (FX)
- ROPE ported from ComfyUI that doesn't use complex numbers. Massive VRAM savings when used with `--compile`\* for inference or `--optimized_compile`\* for training! (`--rope_func comfy`) (WV) (T)
- Optional extra latent noise for I2V/V2V/I2I (`--v2_extra_noise 0.02 --i2_extra_noise 0.02`, values less than 0.04 are recommended. This can improve fine detail and texture in but too much will cause artifacts and moving shadows. I use around 0.01-0.02 for V2V and 0.02-0.04 for I2V) (WV)
- Load mixed precision transformers (`--mixed_precision_transformer` for inference or training, see [here](https://github.com/kohya-ss/musubi-tuner/discussions/232#discussioncomment-13284677) for how to create such a transformer and why you might wanna) (WV) (T)
- Several more LLM options (`--hidden_state_skip_layer N --apply_final_norm`, please see the `--help` for explanations!) (HY)
- FP8 scaled support using the same algo as Wan (`--fp8_scaled`, HIGHLY recommend both for inference and training. It's just better fp8 that's all you need to know!) (HY) (T)
- Separate prompt for CLIP (`--prompt_2 "second prompt goes here"`, provides a different prompt to CLIP since it's used to simpler text) (HY)
- Rescale text encoders based on https://github.com/zer0int/ComfyUI-HunyuanVideo-Nyan (`--te_multiplier llm clip` such as `--te_multiplier 0.9 1.2` to downweight the LLM slightly and upweight the CLIP slightly) (HY)

Non model specific extras:
(Please make sure to install the project into your venv with `--group postprocess` (e.g.`pip install -e . --group postprocess --group dev` to fully install all requirements) if you want to use the below scripts!)
- GIMM-VFI framerate interpolation (`src/blissful_tuner/GIMMVFI.py`, please see it's `--help` for usage. [Models](https://huggingface.co/Blyss/BlissfulModels/tree/main/VFI))
- Upscaling with SwinIR or ESRGAN type models (`src/blissful_tuner/upscaler.py`, please see it's `--help` for usage. [Models](https://huggingface.co/Blyss/BlissfulModels/tree/main/upscaling))
- Face blurring script based on Yolo - helpful for training non face altering LoRA! ( `blissful_tuner/yolo_blur.py`, please see it's `--help` for usage. [Recommended model](https://huggingface.co/Blyss/BlissfulModels/tree/main/yolo))
- Face restoration with CodeFormer/GFPGAN (`src/blissful_tuner/facefix.py`, per usual please have a look at the `--help`! [Models](https://huggingface.co/Blyss/BlissfulModels/tree/main/face_restoration))

(\*) - Features related to torch.compile have additional requirements as well as significant limitations on native Windows platforms so we recommend WSL2 or a native Linux environment instead.

Also a related project of mine called [Envious](https://github.com/Sarania/Envious) is useful for managing Nvidia GPUs from the terminal on Linux. It requires nvidia-ml-py and supports realtime monitoring, over/underclocking, power limit adjustment, fan control, profiles, and more. It also has a little process monitor for the GPU VRAM! Basically it's like nvidia-smi except not bad 😂

My general code and Musubi Tuner code is licensed Apache 2.0. Other projects included may have different licensing, in which case you will find a LICENSE file in their directory specifying the terms under which they are included! Below is the original Musubi Readme which still remains relevant:

# Musubi Tuner Readme

## Table of Contents

<details>
<summary>Click to expand</summary>

- [Musubi Tuner](#musubi-tuner)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Sponsors](#sponsors)
    - [Support the Project](#support-the-project)
    - [Recent Updates](#recent-updates)
    - [Releases](#releases)
    - [For Developers Using AI Coding Agents](#for-developers-using-ai-coding-agents)
  - [Overview](#overview)
    - [Hardware Requirements](#hardware-requirements)
    - [Features](#features)
    - [Documentation](#documentation)
  - [Installation](#installation)
    - [pip based installation](#pip-based-installation)
    - [uv based installation](#uv-based-installation-experimental)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
  - [Model Download](#model-download)
  - [Usage](#usage)
    - [Dataset Configuration](#dataset-configuration)
    - [Pre-caching and Training](#pre-caching-and-training)
    - [Configuration of Accelerate](#configuration-of-accelerate)
    - [Training and Inference](#training-and-inference)
  - [Miscellaneous](#miscellaneous)
    - [SageAttention Installation](#sageattention-installation)
    - [PyTorch version](#pytorch-version)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## Introduction

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo, Wan2.1/2.2, FramePack, FLUX.1 Kontext, and Qwen-Image architectures. 

This repository is unofficial and not affiliated with the official HunyuanVideo/Wan2.1/2.2/FramePack/FLUX.1 Kontext/Qwen-Image repositories.

*This repository is under development.*

### Sponsors

We are grateful to the following companies for their generous sponsorship:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Recent Updates

GitHub Discussions Enabled: We've enabled GitHub Discussions for community Q&A, knowledge sharing, and technical information exchange. Please use Issues for bug reports and feature requests, and Discussions for questions and sharing experiences. [Join the conversation →](https://github.com/kohya-ss/musubi-tuner/discussions)

- December 27, 2025
    - Added support for Qwen-Image-Edit-2511. See [PR #808](https://github.com/kohya-ss/musubi-tuner/pull/808).
        - Please refer to the [documentation](./docs/qwen_image.md) for details such as checkpoints and options.
        - In the caching, training, and inference scripts, specify `--model_version` option as `edit-2511`.

- December 25, 2025
    - Added support for LoRA training of Kandinsky 5. See [PR #774](https://github.com/kohya-ss/musubi-tuner/pull/774). Many thanks to AkaneTendo25 for this contribution.
        - Please refer to the [documentation](./docs/kandinsky5.md) for details.
        - **Note that some weight specifications are in Hugging Face ID format. We plan to change to direct *.safetensors specification like other models soon, so please be aware.**

- December 13, 2025
    - Added support for finetuning Qwen-Image. See [PR #778](https://github.com/kohya-ss/musubi-tuner/pull/778). Many thanks to sdbds for this contribution.
        - Please refer to the [documentation](./docs/zimage.md#finetuning) for details.
    - Added a very simple GUI tool. See [PR #779](https://github.com/kohya-ss/musubi-tuner/pull/779).
        - Currently supports LoRA training for Z-Image-Turbo and Qwen-Image. Please refer to the [documentation](./src/musubi_tuner/gui/gui.md) for details.

- December 9, 2025
    - LoRA weights in Diffusers format can now be loaded with the `--base_weights` option in training scripts. See [PR #772](https://github.com/kohya-ss/musubi-tuner/pull/772).
        - This allows training using Z-Image-Turbo's Training Adapter, etc.
    - Updated the [documentation](./docs/zimage.md) on how to perform LoRA training for Z-Image-Turbo using De-Turbo models or Training Adapters.
    - We would like to express our deep gratitude to ostris for providing these.

- December 7, 2025
    - Added support for Z-Image Turbo. See [PR #757](https://github.com/kohya-ss/musubi-tuner/pull/757).
        - Since this is a Turbo (distilled) model, training may be unstable. Feedback is welcome.
        - Please refer to the [documentation](./docs/zimage.md) for details.

- December 5, 2025
    - Added support for HunyuanVideo 1.5. See [PR #748](https://github.com/kohya-ss/musubi-tuner/pull/748).
        - LoRA training for T2V and I2V is now supported. Please refer to the [documentation](./docs/hunyuan_video_1_5.md) for details.

### Releases

We are grateful to everyone who has been contributing to the Musubi Tuner ecosystem through documentation and third-party tools. To support these valuable contributions, we recommend working with our [releases](https://github.com/kohya-ss/musubi-tuner/releases) as stable reference points, as this project is under active development and breaking changes may occur.

You can find the latest release and version history in our [releases page](https://github.com/kohya-ss/musubi-tuner/releases).

### For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md`, `GEMINI.md`, and/or `AGENTS.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt (currently they are the almost same):

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

    You may be also import the prompt depending on the agent you are using with the custom prompt file such as `AGENTS.md`.

3.  You can now add your own personal instructions below the import line (e.g., `Always include a short summary of the change before diving into details.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md`, `GEMINI.md` and `AGENTS.md` (as well as Claude's `.mcp.json`) are already listed in `.gitignore`, so they won't be committed to the repository.

## Overview

### Hardware Requirements

- VRAM: 12GB or more recommended for image training, 24GB or more for video training
    - *Actual requirements depend on resolution and training settings.* For 12GB, use a resolution of 960x544 or lower and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`, etc.
- Main Memory: 64GB or more recommended, 32GB + swap may work

### Features

- Memory-efficient implementation
- Windows compatibility confirmed (Linux compatibility confirmed by community)
- Multi-GPU training (using [Accelerate](https://huggingface.co/docs/accelerate/index)), documentation will be added later

### Documentation

For detailed information on specific architectures, configurations, and advanced features, please refer to the documentation below.

**Architecture-specific:**
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [Wan2.1/2.2 (Single Frame)](./docs/wan_1f.md)
- [FramePack](./docs/framepack.md)
- [FramePack (Single Frame)](./docs/framepack_1f.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

**Common Configuration & Usage:**
- [Dataset Configuration](./docs/dataset_config.md)
- [Advanced Configuration](./docs/advanced_config.md)
- [Sampling during Training](./docs/sampling_during_training.md)
- [Tools and Utilities](./docs/tools.md)
- [Using torch.compile](./docs/torch_compile.md)

## Installation

### pip based installation

Python 3.10 or later is required (verified with 3.10).

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. 

PyTorch 2.5.1 or later is required (see [note](#PyTorch-version)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command.

```bash
pip install -e .
```

Optionally, you can use FlashAttention and SageAttention (**for inference only**; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Optional dependencies for additional features:
- `ascii-magic`: Used for dataset verification
- `matplotlib`: Used for timestep visualization
- `tensorboard`: Used for logging training progress
- `prompt-toolkit`: Used for interactive prompt editing in Wan2.1 and FramePack inference scripts. If installed, it will be automatically used in interactive mode. Especially useful in Linux environments for easier prompt editing.

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### uv based installation (experimental)

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

Model download procedures vary by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section for instructions.

## Usage


### Dataset Configuration

Please refer to [here](./docs/dataset_config.md).

### Pre-caching

Pre-caching procedures vary by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section for instructions.

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

### Training and Inference

Training and inference procedures vary significantly by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section and the various configuration documents for detailed instructions.

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

This repository is unofficial and not affiliated with the official repositories of the supported architectures. 

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

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Code under the `hunyuan_video_1_5` directory is modified from [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) and follows their license.

Code under the `wan` directory is modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The license is under the Apache License 2.0.

Code under the `frame_pack` directory is modified from [FramePack](https://github.com/lllyasviel/FramePack). The license is under the Apache License 2.0.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
