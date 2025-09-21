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

For architecture-specific documentation, please refer to:
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [FramePack](./docs/framepack.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

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

- September 20, 2025
    - A bug in `qwen_image_generate_image.py` where generation with `--from_file` did not work has been fixed. Thanks to nmfisher for [PR #553](https://github.com/kohya-ss/musubi-tuner/pull/553). Followed by [PR #557](https://github.com/kohya-ss/musubi-tuner/pull/557).
        - Additionally, the `--append_original_name` option has been added to the same script. This appends the base name of the original image to the output file name during editing.

- September 14, 2025
    - A bug was fixed that caused an error when training LoRA for Qwen-Image with `--fp8_base` specified and `--fp8_scaled` not specified using FlashAttention or xformers. See [PR #559](https://github.com/kohya-ss/musubi-tuner/pull/559).
        - However, it is recommended to specify `--fp8_scaled` unless you are running out of memory.

- September 13, 2025
    - A bug in masking during FLF2V inference in `wan_generate_video.py` has been fixed. Thanks to LittleNyima for [PR #548](https://github.com/kohya-ss/musubi-tuner/pull/548).
    - The loading speed of `.safetensors` files has been improved. See [PR #556](https://github.com/kohya-ss/musubi-tuner/pull/556).
        - Model loading can be up to 1.5 times faster.

- September 8, 2025
    - Code analysis with ruff has been introduced, and [contribution guidelines](./CONTRIBUTING.md) have been added.
        - Thanks to arledesma for [Issue #524](https://github.com/kohya-ss/musubi-tuner/issues/524) and [PR #538](https://github.com/kohya-ss/musubi-tuner/pull/538).
    - Activation CPU offloading has been added. See [PR #537](https://github.com/kohya-ss/musubi-tuner/pull/537).
        - This can be used in combination with block swap.
        - This can reduce VRAM usage, especially when training long videos or large batch sizes. Combining it with block swap may enable training that was previously not possible.
        - See the PR and [HunyuanVideo documentation](./docs/hunyuan_video.md#memory-optimization) for details.

- September 6, 2025
    - A new LR scheduler, Rex, has been added. Thanks to xzuyn for [PR #513](https://github.com/kohya-ss/musubi-tuner/pull/513).
        - Similar to the Polynomial Scheduler with power set to less than 1, Rex has a more gradual decrease in learning rate.
        - See [Advanced Configuration documentation](./docs/advanced_config.md#rex) for details.
        
- September 2, 2025 (update)
    - Fine-tuning for Qwen-Image has been added. See [PR #492](https://github.com/kohya-ss/musubi-tuner/pull/492).
        - This trains the entire model rather than just the LoRA layers. See the [finetuning section of the Qwen-Image documentation](./docs/qwen_image.md#finetuning) for details.

- September 2, 2025
    - Code analysis with ruff has been introduced. Thanks to arledesma for [PR #483](https://github.com/kohya-ss/musubi-tuner/pull/483) and [PR #488](https://github.com/kohya-ss/musubi-tuner/pull/488).
        - ruff is a Python code analysis and formatting tool.
    - When contributing code, it would be helpful if you could run `ruff check` to verify the code style. Automatic fixes are also possible with `ruff --fix`.
        - Note that code formatting should be done with `black`, and the `line-length` should be set to `132`.
        - Guidelines will be developed later.

### Releases

We are grateful to everyone who has been contributing to the Musubi Tuner ecosystem through documentation and third-party tools. To support these valuable contributions, we recommend working with our [releases](https://github.com/kohya-ss/musubi-tuner/releases) as stable reference points, as this project is under active development and breaking changes may occur.

You can find the latest release and version history in our [releases page](https://github.com/kohya-ss/musubi-tuner/releases).

### For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt (currently they are the almost same):

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so it won't be committed to the repository.

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

Model download procedures vary by architecture. Please refer to the specific documentation for your chosen architecture:

- [HunyuanVideo model download](./docs/hunyuan_video.md#download-the-model--モデルのダウンロード)
- [Wan2.1/2.2 model download](./docs/wan.md#download-the-model--モデルのダウンロード)
- [FramePack model download](./docs/framepack.md#download-the-model--モデルのダウンロード)
- [FLUX.1 Kontext model download](./docs/flux_kontext.md#download-the-model--モデルのダウンロード)
- [Qwen-Image model download](./docs/qwen_image.md#download-the-model--モデルのダウンロード)

## Usage

### Dataset Configuration

Please refer to [dataset configuration guide](./src/musubi_tuner/dataset/dataset_config.md).

### Pre-caching and Training

Each architecture requires specific pre-caching and training procedures. Please refer to the appropriate documentation:

- [HunyuanVideo usage guide](./docs/hunyuan_video.md)
- [Wan2.1/2.2 usage guide](./docs/wan.md)
- [FramePack usage guide](./docs/framepack.md)
- [FLUX.1 Kontext usage guide](./docs/flux_kontext.md)
- [Qwen-Image usage guide](./docs/qwen_image.md)

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

Training and inference procedures vary significantly by architecture. Please refer to the specific documentation for detailed instructions:

- [HunyuanVideo training and inference](./docs/hunyuan_video.md)
- [Wan2.1/2.2 training and inference](./docs/wan.md)
- [FramePack training and inference](./docs/framepack.md)
- [FLUX.1 Kontext training and inference](./docs/flux_kontext.md)
- [Qwen-Image training and inference](./docs/qwen_image.md)

For advanced configuration options and additional features, refer to:
- [Advanced configuration](./docs/advanced_config.md)
- [Sample generation during training](./docs/sampling_during_training.md)
- [Tools and utilities](./docs/tools.md)

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

Code under the `wan` directory is modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The license is under the Apache License 2.0.

Code under the `frame_pack` directory is modified from [FramePack](https://github.com/lllyasviel/FramePack). The license is under the Apache License 2.0.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.