# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

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

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo, Wan2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2 dev/klein, Qwen-Image, and Z-Image architectures. 

This repository is unofficial and not affiliated with the official repositories of these architectures.

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

- June 16, 2026
    - Added H2D-only block swap, an optimized block swap mode for LoRA (LoHa/LoKr) training, available for all architectures. Enable it with `--block_swap_h2d_only`. See [PR #972](https://github.com/kohya-ss/musubi-tuner/pull/972).
        - For frozen-base training, the base weights on the CPU and GPU are identical, so the classic block swap's device-to-host (D2H) copy is pure overhead. H2D-only keeps a permanent master copy on the CPU and only ever transfers host-to-device, removing the D2H transfer entirely. This can improve training throughput, with the largest benefit when using `--fp8_base` / `--fp8_scaled`.
        - Requires `--gradient_checkpointing`. The number of GPU ring buffers used for streaming can be tuned with `--block_swap_ring_size` (default `2`; `1` minimizes VRAM).
        - There is also a new standalone [Block Swap documentation](./docs/block_swap.md) covering all block swap options.

- June 13, 2026
    - Added the `--save_precision` option for network weights and changed the default save precision to fp32. Thank you rockerBOO [PR #967](https://github.com/kohya-ss/musubi-tuner/pull/967).
        - **Breaking Change**: The default save precision for LoRA files has been changed to fp32.
        - This preserves the precision of LoRA weights during training and is useful for post-processing such as post-hoc EMA, merging, extraction, and weight analysis.
        - LoRA files may be about twice as large as before when training with `--mixed_precision bf16`(`fp16`). To keep the previous behavior, specify `--save_precision bf16` (`fp16`).
        - Please refer to the [HunyuanVideo documentation](./docs/hunyuan_video.md#training--学習) for details.

- June 8, 2026
    - Added experimental support for HiDream-O1-Image (LoRA training, full finetuning, and inference). See [PR #964](https://github.com/kohya-ss/musubi-tuner/pull/964).
        - Please refer to the [documentation](./docs/hidream_o1.md) for details.
        - An optional DINOv3 auxiliary perceptual loss is also available. See the [advanced configuration documentation](./docs/advanced_config.md).
        - Many thanks to sdbds for [PR #947](https://github.com/kohya-ss/musubi-tuner/pull/947) (followed by [PR #955](https://github.com/kohya-ss/musubi-tuner/pull/955)), which this support is based on. Please open the PRs if you would like to review the changes in detail.

- May 22, 2026
    - Performed a large-scale internal refactoring to improve code quality and maintainability. See [PR #950](https://github.com/kohya-ss/musubi-tuner/pull/950)
        - We have taken care to ensure that there are no direct impacts on users. For details and to report any issues, please refer to [this discussion](https://github.com/kohya-ss/musubi-tuner/discussions/949).

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
- [Z-Image](./docs/zimage.md)
- [HiDream-O1-Image](./docs/hidream_o1.md)
- [HunyuanVideo 1.5](./docs/hunyuan_video_1_5.md)
- [Kandinsky 5](./docs/kandinsky5.md)
- [FLUX.2](./docs/flux_2.md)

**Common Configuration & Usage:**
- [Dataset Configuration](./docs/dataset_config.md)
- [Advanced Configuration](./docs/advanced_config.md)
- [Sampling during Training](./docs/sampling_during_training.md)
- [Block Swap (CPU Offloading for Memory Saving)](./docs/block_swap.md)
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
