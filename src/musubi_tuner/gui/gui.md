[English](./gui.md) | [日本語](./gui.ja.md)

# Musubi Tuner GUI - User Guide

This guide will help you set up and use the Musubi Tuner GUI for training LoRA models with image generation architectures like Z-Image-Turbo and Qwen-Image.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing Required Software](#installing-required-software)
3. [Installing Musubi Tuner](#installing-musubi-tuner)
4. [Launching the GUI](#launching-the-gui)
5. [Workflow Guide](#workflow-guide)
6. [Field Descriptions](#field-descriptions)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, make sure you have:

- A Windows PC with an NVIDIA GPU (12GB+ VRAM recommended, 64GB+ VRAM + main RAM recommended)
- Internet connection
- ComfyUI installed with the required models (VAE, Text Encoder, DiT)

---

## Installing Required Software

### Step 1: Install Python

Musubi Tuner requires Python 3.10, 3.11, or 3.12. If you don't have it installed, follow these steps:

1. Go to the official Python website: https://www.python.org/downloads/
2. Download Python 3.10, 3.11, or 3.12 Windows 64-bit installer (we recommend **Python 3.12** for best compatibility)
3. Run the installer
4. **IMPORTANT**: Check the box that says **"Add Python to PATH"** before clicking "Install Now"
5. Complete the installation

**Verify installation**: Open Command Prompt (press `Win + R`, type `cmd`, press Enter) and run:
```
python --version
```
You should see something like `Python 3.12.x`.

### Step 2: Install Git

Git is needed to download the Musubi Tuner source code. If you don't have it installed, follow these steps:

1. Go to the Git website: https://git-scm.com/downloads/win
2. Download the Windows installer
3. Run the installer with default settings (keep clicking "Next")
4. Complete the installation

**Verify installation**: In Command Prompt, run:
```
git --version
```
You should see something like `git version 2.x.x`.

### Step 3: Install uv

uv is a modern Python package manager that simplifies dependency management.

1. Open Command Prompt as Administrator
2. Run the following command:
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
3. **Close and reopen** Command Prompt (normal, non-administrator) for the changes to take effect

**Verify installation**: In a new Command Prompt, run:
```
uv --version
```
You should see something like `uv 0.x.x`.

---

## Installing Musubi Tuner

### Step 1: Download the Source Code

1. Open Command Prompt
2. Navigate to a folder where you want to install Musubi Tuner. For example:
   ```
   cd C:\Users\YourName\Documents
   ```
3. Clone the repository:
   ```
   git clone https://github.com/kohya-ss/musubi-tuner.git
   ```
4. Navigate into the folder:
   ```
   cd musubi-tuner
   ```

### Step 2: First-Time Setup (Automatic)

The first time you run the GUI, uv will automatically download and install all required dependencies including PyTorch. This may take several minutes.

---

## Launching the GUI

Open Command Prompt, navigate to the musubi-tuner folder, and run one of the following commands based on your CUDA version:

### For CUDA 12.4 (Stable version)

```
uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
```

### For CUDA 12.8 (Newer GPUs)

```
uv run --extra cu128 --extra gui python src/musubi_tuner/gui/gui.py
```

**Note**: If you're unsure which CUDA version to use, try `cu124` first.

After launching, the GUI will start and display a URL like:
```
Running on local URL:  http://127.0.0.1:7860
```

Open this URL in your web browser to access the GUI.

**Tip**: You can create a batch file (`.bat`) to launch the GUI more easily:

1. Create a file named `launch_gui.bat` in the `musubi-tuner` folder
2. Add the following content:
   ```batch
   @echo off
   cd /d "%~dp0"
   uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
   pause
   ```
3. Double-click the batch file to launch the GUI

---

## Workflow Guide

The GUI is organized from top to bottom in the order you should complete each step.

### Overview of Steps

1. **Project Setup** - Create or open a project folder
2. **Model Selection** - Choose the model architecture and set up model paths
3. **Dataset Configuration** - Configure training resolution and batch size
4. **Preprocessing** - Cache latents and text encoder outputs
5. **Training** - Configure and start the training process
6. **Post-Processing** - Convert LoRA for ComfyUI (if needed)

---

### Step 1: Project Setup

1. **Project Directory**: Enter the full path to your project folder (e.g., `C:\MyProjects\my_lora_project`)
2. Click **"Initialize / Load Project"**

This will:
- Create the project folder if it doesn't exist
- Create a `training` subfolder for your training images
- Load previous settings if the project was used before

**After initialization**, place your training data in the `training` folder:
- Image files (`.jpg`, `.png`, etc.)
- Caption files (same filename as the image, but with `.txt` extension)

Example:
```
my_lora_project/
  training/
    image001.jpg
    image001.txt
    image002.png
    image002.txt
```

---

### Step 2: Model Selection

1. **Model Architecture**: Select the model you want to train
   - `Z-Image-Turbo` - Faster training; LoRA training may be slightly unstable because the Base model is not released yet
   - `Qwen-Image` - Higher quality, requires more VRAM

2. **VRAM Size**: Select your GPU's VRAM size
   - This affects recommended settings like batch size and block swap

3. **ComfyUI Models Directory**: Enter the path to your ComfyUI `models` folder
   - Example: `C:\ComfyUI\models`
   - This folder should contain `vae`, `text_encoders`, and `diffusion_models` subfolders
   - Required models can be found in the [Required Model Files](#required-model-files) section below

4. Click **"Validate ComfyUI Models Directory"** to verify the folder structure

---

### Step 3: Dataset Configuration

1. Click **"Set Recommended Resolution & Batch Size"** to auto-fill recommended values for your selected model and VRAM
2. Adjust if needed:
   - **Resolution (Width/Height)**: Training image resolution
   - **Batch Size**: Number of images processed at once (higher = faster but more VRAM)
3. Click **"Generate Dataset Config"** to create the configuration file

The generated configuration will appear in the preview area below the button.

---

### Step 4: Preprocessing

Before training, you need to cache the latents and text encoder outputs. This converts your images and captions into a format the model can use.

1. Click **"Set Default Paths"** to auto-fill model paths based on your ComfyUI directory
2. Verify the paths are correct:
   - **VAE Path**: Path to the VAE model
   - **Text Encoder 1 Path**: Path to the text encoder model
   - **Text Encoder 2 Path**: (Only for some models, may be empty)

3. Click **"Cache Latents"** and wait for it to complete
   - This encodes your images into latent space
   - Watch the log output for progress

4. Click **"Cache Text Encoder Outputs"** and wait for it to complete
   - This encodes your captions into embeddings
   - This may take a while for the first run as the text encoder is loaded

---

### Step 5: Training

1. Click **"Set Recommended Parameters"** to auto-fill training settings for your model and VRAM

2. **Configure Required Settings**:
   - **Base Model / DiT Path**: Path to the diffusion model (auto-filled if you click the recommended button)
   - **Output Name**: Name for your LoRA file (e.g., `my_character_lora`)

3. **Basic Parameters** (can use defaults):
   - **LoRA Dim**: LoRA rank/dimension (4-32, higher = more capacity but larger file)
   - **Learning Rate**: How fast the model learns (default: 1e-3 (0.001), can decrease if training is unstable)
   - **Epochs**: Number of times to train on all images. Default is adjusted based on image count; reduce if overfitting occurs.
   - **Save Every N Epochs**: How often to save checkpoints

4. **Advanced Parameters** (expand "Advanced Parameters" accordion):
   - **Discrete Flow Shift**: Which denoising step to emphasize (model-specific defaults recommended)
   - **Block Swap**: Offloads model layers to CPU (use if VRAM is limited)
   - **Mixed Precision**: Precision mode (bf16 recommended)
   - **Gradient Checkpointing**: Reduces VRAM usage
   - **FP8 options**: Further memory optimization

5. **Sample Image Generation** (optional):
   - Enable **"Generate sample images during training"** to see progress
   - Enter a sample prompt that represents what you're training
   - Set the sample image size and frequency

6. Click **"Start Training"** to begin
   - A new command window will open showing training progress
   - Training progress is displayed in the new window
   - The GUI will show a message confirming training has started

---

### Step 6: Post-Processing (Optional)

Z-Image LoRAs need to be converted for use in ComfyUI. Follow these steps:

1. Click **"Set Default Paths"** to auto-fill paths based on your output name
2. Verify the paths:
   - **Input LoRA Path**: Path to your trained LoRA
   - **Output ComfyUI LoRA Path**: Where to save the converted LoRA
3. Click **"Convert to ComfyUI Format"**

---

## Required Model Files

### Z-Image-Turbo

For text encoder and VAE model files, download them from the appropriate directory under `split_files` here: https://huggingface.co/Comfy-Org/z_image_turbo

| Type | Model file |
|------|------------|
| diffusion-models | Use `z_image_de_turbo_v1_bf16.safetensors` from ostris's [De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) |
| text-encoders | `qwen_3_4b.safetensors` |
| VAE | `ae.safetensors` |

### Qwen-Image

Download the required model files from the appropriate directory under `split_files` here: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI

| Type | Model file |
|------|------------|
| diffusion-models | `qwen_image_bf16.safetensors` |
| text-encoders | `qwen_2.5_vl_7b.safetensors` |
| VAE | `qwen_image_vae.safetensors` |

---

## Field Descriptions

### Project Settings

| Field | Description |
|-------|-------------|
| Project Directory | Root folder for your training project. All data and outputs will be stored here. |

### Model Settings

| Field | Description |
|-------|-------------|
| Model Architecture | The base model to train a LoRA for. Z-Image-Turbo is faster; Qwen-Image produces higher quality. |
| VRAM Size | Your GPU's video memory. Affects recommended batch size and memory optimization settings. |
| ComfyUI Models Directory | Path to ComfyUI's `models` folder containing the required model files. |

### Dataset Settings

| Field | Description |
|-------|-------------|
| Resolution (W/H) | Training resolution. Images will be resized/cropped to this size. |
| Batch Size | Number of images processed simultaneously. Higher values train faster but use more VRAM. |

### Preprocessing

| Field | Description |
|-------|-------------|
| VAE Path | Path to the VAE model file used to encode images into latent space. |
| Text Encoder 1 Path | Path to the main text encoder model. |
| Text Encoder 2 Path | Path to secondary text encoder (if required by the model). |

### Training Parameters

| Field | Description |
|-------|-------------|
| Base Model / DiT Path | Path to the base diffusion model (DiT). |
| Output Name | Base name for saved LoRA files (without extension). |
| LoRA Dim | LoRA rank/dimension. Higher values capture more detail but create larger files. Common values: 4, 8, 16, 32. |
| Learning Rate | Speed of training. Higher = faster learning but may overshoot. Default: 1e-3 (0.001). |
| Epochs | Number of complete passes through all training images. |
| Save Every N Epochs | Frequency of checkpoint saves. Also controls sample image generation frequency. |
| Discrete Flow Shift | Flow matching parameter that affects training dynamics. Model-specific defaults are recommended. |
| Block Swap | Number of transformer blocks to offload to CPU. Use when VRAM is limited. Higher = less VRAM but slower. |
| Mixed Precision | Floating-point precision. bf16 recommended for modern GPUs. |
| Gradient Checkpointing | Reduces VRAM usage by recomputing some values. Slightly slower but uses less memory. |
| FP8 Scaled | Use FP8 precision for the base model. Reduces memory with minimal quality loss. |
| FP8 LLM | Use FP8 precision for the text encoder (LLM). Further reduces memory usage. |
| Additional Arguments | Extra command-line arguments for advanced users. |

### Sample Image Generation

| Field | Description |
|-------|-------------|
| Generate sample images | Enable to generate sample images during training. |
| Sample Prompt | Text prompt used to generate sample images. |
| Negative Prompt | What to avoid in sample images. |
| Sample Width/Height | Resolution for sample images. |
| Sample Every N Epochs | How often to generate samples. |

### Post-Processing

| Field | Description |
|-------|-------------|
| Input LoRA Path | Path to the trained LoRA file (in Musubi Tuner format). |
| Output ComfyUI LoRA Path | Where to save the converted LoRA (in ComfyUI format). |

---

## Troubleshooting

### "Python is not recognized"
- Make sure you checked "Add Python to PATH" during installation
- Try reinstalling Python with this option enabled
- Or manually add Python to your system PATH

### "uv is not recognized"
- Close and reopen Command Prompt after installing uv
- Try running the installation command again

### CUDA errors or out of memory
- Select a smaller VRAM size in the GUI to get more conservative settings
- Enable Block Swap to offload some computation to CPU
- Reduce batch size to 1
- Enable FP8 options for additional memory savings

### Training script exits with errors immediately
- Check the error message for clues
- Check if all paths are correct
- Make sure preprocessing (Cache Latents and Cache Text Encoder) completed successfully

### Slow training
- If Block Swap is enabled, training will be slower (this is expected when VRAM is limited)
- If VRAM is insufficient and shared VRAM is being used, performance will degrade significantly. Try reducing memory usage by using FP8 options, increasing Block Swap, or lowering batch size.
- Make sure you're using a GPU (not CPU)
- Check that your GPU drivers are up to date

### "Model not found" errors
- Verify that your ComfyUI models directory is correct
- Make sure you have downloaded the required models
- Check that the model filenames match what the GUI expects (see config_manager.py for exact filenames)

### GUI won't start
- Make sure you're in the correct directory (musubi-tuner folder)
- Make sure you're using the correct uv command (check your CUDA version)

---

## Project Folder Structure

After using the GUI, your project folder will look like this:

```
my_lora_project/
  training/           # Your training images and captions
    image001.jpg
    image001.txt
    ...
  cache/              # Preprocessed data (auto-created)
    latent_cache/
    text_encoder_cache/
  models/             # Trained LoRA files (auto-created)
    my_lora.safetensors
    my_lora_comfy.safetensors
    sample/           # Sample images generated during training
  logs/               # TensorBoard logs (auto-created)
  dataset_config.toml # Dataset configuration (auto-created)
  musubi_project.toml # GUI Project settings (auto-created)
  sample_prompt.txt   # Sample prompt file (auto-created if enabled)
```

---

## Next Steps

After training your LoRA:

1. If you need to use it in ComfyUI (Z-Image LoRAs need conversion), convert it using the Post-Processing section
2. Copy the converted LoRA to your ComfyUI `models/loras` folder
3. Load it in ComfyUI using a LoRA loader node

For more advanced training options and command-line usage, refer to the main Musubi Tuner documentation in the `docs` folder.
