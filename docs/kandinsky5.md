> 📝 Click on the language section to expand / 言語をクリックして展開

# Kandinsky 5

## Overview / 概要

This is an unofficial training and inference script for [Kandinsky 5](https://github.com/ai-forever/Kandinsky-5). The features are as follows:

- fp8 support and memory reduction by block swap
- Inference without installing Flash attention (using PyTorch's scaled dot product attention)
- LoRA training for text-to-video (T2V) models

This feature is experimental.

<details>
<summary>日本語</summary>

[Kandinsky 5](https://github.com/ai-forever/Kandinsky-5) の非公式の学習および推論スクリプトです。

以下の特徴があります：

- fp8対応およびblock swapによる省メモリ化
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- テキストから動画（T2V）モデルのLoRA学習

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

Download the model weights from the [Kandinsky 5.0 Collection](https://huggingface.co/collections/ai-forever/kandinsky-50) on Hugging Face.

### DiT Model / DiTモデル

Download the DiT checkpoint from one of the following repositories:

**Lite models (2B parameters):**
- [Kandinsky-5.0-T2V-Lite-sft-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s) - SFT model, highest quality
- [Kandinsky-5.0-T2V-Lite-sft-10s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-10s) - 10 second videos
- [Kandinsky-5.0-T2V-Lite-pretrain-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-pretrain-5s) - Pretrain model for fine-tuning
- [Kandinsky-5.0-T2V-Lite-nocfg-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-nocfg-5s) - CFG-distilled, 2x faster
- [Kandinsky-5.0-T2V-Lite-distilled16steps-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-5s) - Diffusion-distilled, 6x faster
- [Kandinsky-5.0-I2V-Lite-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s) - Image-to-Video

Download the `.safetensors` file (e.g., `kandinsky5lite_t2v_sft_5s.safetensors`).

### VAE

Kandinsky 5 uses the HunyuanVideo 3D VAE. Download `diffusion_pytorch_model.safetensors` (or `pytorch_model.pt`) from:
https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae

### Text Encoders / テキストエンコーダ

Kandinsky 5 uses Qwen2.5-VL and CLIP for text encoding.

**Qwen2.5-VL**: Download from https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct (or use the path to your local Qwen2.5-VL model)

**CLIP**: Download `clip_l.safetensors` from https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders and place it in a directory (e.g., `text_encoder2/clip_l.safetensors`)

### Directory Structure / ディレクトリ構造

Place them in your chosen directory structure:

```
weights/
├── model/
│   └── kandinsky5lite_t2v_sft_5s.safetensors
├── vae/
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   └── (Qwen2.5-VL files)
└── text_encoder2/
    └── clip_l.safetensors
```

<details>
<summary>日本語</summary>

Hugging Faceの[Kandinsky 5.0 Collection](https://huggingface.co/collections/ai-forever/kandinsky-50)からモデルの重みをダウンロードしてください。

**DiTモデル**: 上記のリポジトリから`.safetensors`ファイルをダウンロードしてください。

**VAE**: Kandinsky 5はHunyuanVideo 3D VAEを使用します。上記リンクから`diffusion_pytorch_model.safetensors`（または`pytorch_model.pt`）をダウンロードしてください。

**テキストエンコーダ**: Qwen2.5-VLとCLIPを使用します。上記リンクからダウンロードしてください。

任意のディレクトリ構造に配置してください。

</details>

## Available Tasks / 利用可能なタスク

The `--task` option specifies the model configuration. Available tasks:

**Lite models (2B parameters):**

| Task | Description | Resolution | Notes |
|------|-------------|------------|-------|
| `k5-lite-t2i-hd` | Lite T2I | 1024 | Image generation |
| `k5-lite-i2i-hd` | Lite I2I | 1024 | Image-to-image |
| `k5-lite-t2v-5s-sd` | Lite T2V 5s | 512 | SFT model |
| `k5-lite-t2v-10s-sd` | Lite T2V 10s | 512 | SFT model |
| `k5-lite-i2v-5s-sd` | Lite I2V 5s | 512 | Image-to-video |
| `k5-lite-t2v-5s-distil-sd` | Lite T2V 5s Distilled | 512 | 16 steps, faster |
| `k5-lite-t2v-10s-distil-sd` | Lite T2V 10s Distilled | 512 | 16 steps, faster |
| `k5-lite-t2v-5s-nocfg-sd` | Lite T2V 5s No-CFG | 512 | CFG-distilled |
| `k5-lite-t2v-10s-nocfg-sd` | Lite T2V 10s No-CFG | 512 | CFG-distilled |
| `k5-lite-t2v-5s-pretrain-sd` | Lite T2V 5s Pretrain | 512 | For fine-tuning |
| `k5-lite-t2v-10s-pretrain-sd` | Lite T2V 10s Pretrain | 512 | For fine-tuning |

**Pro models (19B parameters):**

| Task | Description | Resolution |
|------|-------------|------------|
| `k5-pro-t2v-5s-sd` | Pro T2V 5s SD | 512 |
| `k5-pro-t2v-5s-hd` | Pro T2V 5s HD | 1024 |
| `k5-pro-t2v-10s-sd` | Pro T2V 10s SD | 512 |
| `k5-pro-t2v-10s-hd` | Pro T2V 10s HD | 1024 |
| `k5-pro-i2v-5s-sd` | Pro I2V 5s SD | 512 |
| `k5-pro-i2v-5s-hd` | Pro I2V 5s HD | 1024 |

<details>
<summary>日本語</summary>

`--task`オプションでモデル設定を指定します。利用可能なタスクは上記の表を参照してください。

**Liteモデル (2Bパラメータ)**: 公開されている軽量モデルです。

**Proモデル (19Bパラメータ)**: 高品質な大規模モデルです。

</details>

## Pre-caching / 事前キャッシュ

Pre-caching is required before training. This involves caching both latents and text encoder outputs.

### Text Encoder Output Pre-caching / テキストエンコーダ出力の事前キャッシュ

Text encoder output pre-caching is required. Create the cache using the following command:

```bash
python kandinsky5_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --text_encoder_qwen path/to/text_encoder \
    --text_encoder_clip path/to/text_encoder2 \
    --batch_size 4
```

Adjust `--batch_size` according to your available VRAM.

For additional options, use `python kandinsky5_cache_text_encoder_outputs.py --help`.

<details>
<summary>日本語</summary>

テキストエンコーダ出力の事前キャッシュは必須です。上のコマンド例を使用してキャッシュを作成してください。

使用可能なVRAMに合わせて `--batch_size` を調整してください。

その他のオプションは `--help` で確認できます。

</details>

### Latent Pre-caching / latentの事前キャッシュ

Latent pre-caching is required. Create the cache using the following command:

```bash
python kandinsky5_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae path/to/vae/diffusion_pytorch_model.safetensors
```

If you're running low on VRAM, lower the `--batch_size`.

For additional options, use `python kandinsky5_cache_latents.py --help`.

<details>
<summary>日本語</summary>

latentの事前キャッシュは必須です。上のコマンド例を使用してキャッシュを作成してください。

VRAMが足りない場合は、`--batch_size`を小さくしてください。

その他のオプションは `--help` で確認できます。

</details>

## Training / 学習

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    kandinsky5_train_network.py \
    --mixed_precision bf16 \
    --dataset_config path/to/dataset.toml \
    --task k5-pro-t2v-5s-sd \
    --dit path/to/kandinsky5pro_t2v_pretrain_5s.safetensors \
    --text_encoder_qwen path/to/text_encoder \
    --text_encoder_clip path/to/text_encoder2 \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --fp8_base \
    --blocks_to_swap 10 \
    --flash_attn \
    --gradient_checkpointing \
    --max_data_loader_n_workers 1 \
    --persistent_data_loader_workers \
    --learning_rate 1e-4 \
    --optimizer_type AdamW8Bit \
    --optimizer_args "weight_decay=0.001" "betas=(0.9,0.95)" \
    --max_grad_norm 1.0 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 100 \
    --network_module networks.lora_kandinsky \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 5.0 \
    --output_dir path/to/output \
    --output_name my_lora \
    --save_every_n_epochs 5 \
    --max_train_epochs 50 \
    --scheduler_scale 10.0
```

The training settings are experimental. Appropriate learning rates, training steps, timestep distribution, etc. are not yet fully determined. Feedback is welcome.

For additional options, use `python kandinsky5_train_network.py --help`.

### Key Options / 主要オプション

- `--task`: Model configuration (architecture, attention type, resolution, sampling parameters). See Available Tasks above.
- `--dit`: Path to DiT checkpoint. **Overrides the task's default checkpoint path.** You can use any compatible checkpoint (SFT, pretrain, or your own) with any task config as long as the architecture matches.
- `--vae`: Path to VAE checkpoint (overrides task default)
- `--network_module`: Use `networks.lora_kandinsky` for Kandinsky5 LoRA

**Note**: The `--task` option only sets the model architecture and parameters, not the weights. Use `--dit` to specify which checkpoint to load. For example, you can train a pretrain checkpoint using `--task k5-lite-t2v-5s-sd --dit path/to/pretrain.safetensors`.

**注意**: `--task`オプションはモデルのアーキテクチャとパラメータのみを設定し、重みは設定しません。`--dit`で読み込むチェックポイントを指定してください。例えば、`--task k5-lite-t2v-5s-sd --dit path/to/pretrain.safetensors`のように、pretrainチェックポイントを使用できます。

### Memory Optimization / メモリ最適化

`--gradient_checkpointing` enables gradient checkpointing to reduce VRAM usage.

`--fp8_base` runs DiT in fp8 mode. This can significantly reduce memory consumption but may impact output quality.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU.

`--gradient_checkpointing_cpu_offload` can be used to offload activations to CPU when using gradient checkpointing. This must be used together with `--gradient_checkpointing`.

### Attention / アテンション

Use `--sdpa` for PyTorch's scaled dot product attention. Use `--flash_attn` for FlashAttention. Use `--xformers` for xformers.

### Timestep Sampling / タイムステップサンプリング

You can specify the range of timesteps with `--min_timestep` and `--max_timestep`. See [advanced configuration](./advanced_config.md) for details.

### Sample Generation During Training / 学習中のサンプル生成

Sample generation during training is supported. See [sampling during training](./sampling_during_training.md) for details.

<details>
<summary>日本語</summary>

上のコマンド例を使用して学習を開始してください（実際には一行で入力）。

学習設定は実験的なものです。適切な学習率、学習ステップ数、タイムステップの分布などは、まだ完全には決まっていません。フィードバックをお待ちしています。

その他のオプションは `--help` で確認できます。

**主要オプション**

- `--task`: モデル設定（上記の利用可能なタスクを参照）
- `--dit`: DiTチェックポイントへのパス（タスクのデフォルトを上書き）
- `--vae`: VAEチェックポイントへのパス（タスクのデフォルトを上書き）
- `--network_module`: Kandinsky5 LoRAには `networks.lora_kandinsky` を使用

**メモリ最適化**

`--gradient_checkpointing`でgradient checkpointingを有効にし、VRAM使用量を削減できます。

`--fp8_base`を指定すると、DiTがfp8で学習されます。消費メモリを大きく削減できますが、品質は低下する可能性があります。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。

`--gradient_checkpointing_cpu_offload`を指定すると、gradient checkpointing使用時にアクティベーションをCPUにオフロードします。`--gradient_checkpointing`と併用する必要があります。

**アテンション**

`--sdpa`でPyTorchのscaled dot product attentionを使用します。`--flash_attn`でFlashAttentionを使用します。`--xformers`でxformersを使用します。

**タイムステップサンプリング**

`--min_timestep`と`--max_timestep`を指定すると、学習時のタイムステップの範囲を指定できます。詳細は[高度な設定](./advanced_config.md)を参照してください。

**学習中のサンプル生成**

学習中のサンプル生成がサポートされています。詳細は[学習中のサンプリング](./sampling_during_training.md)を参照してください。

</details>

## Inference / 推論

Generate videos using the following command:

```bash
python kandinsky5_generate_video.py \
    --task k5-pro-t2v-5s-sd \
    --dit path/to/kandinsky5pro_t2v_pretrain_5s.safetensors \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --text_encoder_qwen path/to/text_encoder \
    --text_encoder_clip path/to/text_encoder2 \
    --blocks_to_swap 10 \
    --offload_dit_during_sampling \
    --fp8_base \
    --dtype bfloat16 \
    --prompt "A cat walks on the grass, realistic style." \
    --negative_prompt "low quality, artifacts" \
    --frames 17 \
    --steps 50 \
    --guidance 5 \
    --scheduler_scale 10 \
    --seed 42 \
    --width 512 \
    --height 512 \
    --output path/to/output.mp4 \
    --lora_weight path/to/lora.safetensors \
    --lora_multiplier 1.0
```

### Options / オプション

- `--task`: Model configuration
- `--prompt`: Text prompt for generation
- `--negative_prompt`: Negative prompt (optional)
- `--output`: Output file path (.mp4 for video, .png for image)
- `--width`, `--height`: Output resolution (defaults from task config)
- `--frames`: Number of frames (defaults from task config)
- `--steps`: Number of inference steps (defaults from task config)
- `--guidance`: Guidance scale (defaults from task config)
- `--seed`: Random seed
- `--fp8_base`: Run DiT in fp8 mode
- `--blocks_to_swap`: Number of blocks to offload to CPU
- `--lora_weight`: Path(s) to LoRA weight file(s)
- `--lora_multiplier`: LoRA multiplier(s)

For additional options, use `python kandinsky5_generate_video.py --help`.

<details>
<summary>日本語</summary>

上のコマンド例を使用して動画を生成します。

**オプション**

- `--task`: モデル設定
- `--prompt`: 生成用のテキストプロンプト
- `--negative_prompt`: ネガティブプロンプト（オプション）
- `--output`: 出力ファイルパス（動画は.mp4、画像は.png）
- `--width`, `--height`: 出力解像度（タスク設定からのデフォルト）
- `--frames`: フレーム数（タスク設定からのデフォルト）
- `--steps`: 推論ステップ数（タスク設定からのデフォルト）
- `--guidance`: ガイダンススケール（タスク設定からのデフォルト）
- `--seed`: ランダムシード
- `--fp8_base`: DiTをfp8モードで実行
- `--blocks_to_swap`: CPUにオフロードするブロック数
- `--lora_weight`: LoRA重みファイルへのパス
- `--lora_multiplier`: LoRA係数

その他のオプションは `--help` で確認できます。

</details>

## FP8 Checkpoints / FP8チェックポイント

You can pre-quantize the DiT checkpoint to fp8 format using the provided script:

```bash
python _make_fp8_ckpt.py \
    --input path/to/kandinsky5pro_t2v_sft_5s.safetensors \
    --output path/to/kandinsky5pro_t2v_sft_5s_fp8.safetensors
```

The fp8 checkpoint can be used directly with training and inference scripts. When an fp8 checkpoint is detected, it will be used as-is without re-quantization.

<details>
<summary>日本語</summary>

提供されているスクリプトを使用して、DiTチェックポイントをfp8形式に事前量子化できます。

fp8チェックポイントは、学習および推論スクリプトで直接使用できます。fp8チェックポイントが検出されると、再量子化せずにそのまま使用されます。

</details>

## Dataset Configuration / データセット設定

Dataset configuration is the same as other architectures. See [dataset configuration](./dataset_config.md) for details.

```toml
[general]
enable_bucket = true
bucket_no_upscale = false

# Image dataset example
[[datasets]]
image_directory = "path/to/images"
cache_directory = "path/to/images/cache"
resolution = [512, 512]
batch_size = 1
num_repeats = 1
caption_extension = ".txt"

# Video dataset example
[[datasets]]
video_directory = "path/to/videos"
cache_directory = "path/to/videos/cache"
resolution = [256, 256]
batch_size = 1
num_repeats = 1
frame_extraction = "head"
target_frames = [17]
caption_extension = ".txt"
```

Note: `target_frames` values must follow the `N*4+1` pattern (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, ...).

<details>
<summary>日本語</summary>

データセット設定は他のアーキテクチャと同じです。詳細は[データセット設定](./dataset_config.md)を参照してください。

データセットTOMLの形式は他のアーキテクチャと同じです。

注意: `target_frames` の値は `N*4+1` パターン（1, 5, 9, 13, 17, 21, 25, 29, 33, ...）に従う必要があります。

</details>
