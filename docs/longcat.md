> 📝 Click on the language section to expand / 言語をクリックして展開

# LongCat

## Overview / 概要

LongCat-Video is a 13.6B parameter video generation model that natively covers text-to-video, image-to-video, and video-continuation tasks. Musubi Tuner adds LoRA training and inference utilities while reusing the same WanVAE and UMT5 encoder (tokenizer bundled inside the T5 weights).

- Uses WanVAE and the shared UMT5 encoder; no separate tokenizer setup is required
- Supports the same attention backends as other Musubi Tuner video trainers (FlashAttention 2/3, xFormers, torch SDPA)
- Provides text-to-video, image-to-video, and refinement entry points aligned with Musubi Tuner scripts

This integration is experimental.

<details>
<summary>日本語</summary>

LongCat-Video は 135 億パラメータ規模の動画生成モデルで、テキスト→動画、画像→動画、ビデオ継続を単一チェックポイントで扱えます。Musubi Tuner では WanVAE と共用の UMT5 エンコーダ（トークナイザは T5 重みに内蔵）をそのまま使用し、LoRA 学習・推論用のスクリプトを提供します。

- WanVAE と共用 UMT5 エンコーダを利用するため追加のトークナイザ設定は不要
- 他の動画トレーナー同様に FlashAttention 2/3、xFormers、torch SDPA を選択可能
- Musubi Tuner 標準の T2V / I2V / リファイン入口を備えています

本統合は実験的機能です。

</details>

## Download the model / モデルのダウンロード

Collect the following weights (mirroring the official checkpoint layout):

1. **Text Encoder** – UMT5 encoder weights (e.g. `models_t5_umt5-xxl-enc-bf16.pth`). Musubi Tuner loads the embedded tokenizer directly from this file via `--text_encoder`.
2. **VAE** – Wan VAE (`Wan2.1_VAE.pth` or `wan_2.1_vae.safetensors`).
3. **LongCat DiT** – LongCat `.safetensors` checkpoint (e.g. `longcat_video-13B.safetensors`).

Keep files organized, e.g.:

```
ckpts/
  longcat/
    longcat_dit.safetensors
    Wan2.1_VAE.pth
    models_t5_umt5-xxl-enc-bf16.pth
```

<details>
<summary>日本語</summary>

公式チェックポイントと同じ構成で重みを準備してください。

1. **テキストエンコーダ** – UMT5 重み (`models_t5_umt5-xxl-enc-bf16.pth` など)。`--text_encoder` で指定すると内蔵トークナイザが利用されます。
2. **VAE** – Wan VAE (`Wan2.1_VAE.pth` または `wan_2.1_vae.safetensors`)。
3. **LongCat DiT 重み** – LongCat 用 `.safetensors` チェックポイント（例: `longcat_video-13B.safetensors`）。

以下のように配置すると便利です。

```
ckpts/
  longcat/
    longcat_dit.safetensors
    Wan2.1_VAE.pth
    models_t5_umt5-xxl-enc-bf16.pth
```

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latent の事前キャッシュ

LongCat shares the WAN VAE interface, so you can cache latents before training:

```bash
python src/musubi_tuner/longcat_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --vae_cache_cpu
```

- Use `--i2v` when preparing datasets with conditioning images.
- `--vae_cache_cpu` keeps WanVAE feature maps in system RAM to save VRAM.

### Text Encoder Output Pre-caching / テキストエンコーダ出力の事前キャッシュ

```bash
python src/musubi_tuner/longcat_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 8 \
    --fp8_t5
```

- `--fp8_t5` reduces memory usage when running the WAN T5 weights.
- Outputs are reused by the training script for faster iterations.

<details>
<summary>日本語</summary>

LongCat は WAN VAE を流用しているため、学習前に latent をキャッシュできます。

```bash
python src/musubi_tuner/longcat_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --vae_cache_cpu
```

- 画像条件付きデータセットでは `--i2v` を追加してください。
- `--vae_cache_cpu` で VAE フィーチャマップを CPU に退避し、VRAM を節約します。

テキストエンコーダ出力も事前キャッシュが可能です。

```bash
python src/musubi_tuner/longcat_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --batch_size 8 \
    --fp8_t5
```

- `--fp8_t5` を指定すると WAN T5 を fp8 で動作させ、メモリ使用量を削減できます。
- 出力は学習時に再利用され、イテレーションが高速化されます。

</details>

## Training / 学習

Train LongCat LoRA networks via `longcat_train_network.py`, which extends the WAN trainer with LongCat-specific defaults.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/longcat_train_network.py \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dataset_config path/to/dataset.toml \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --blocks_to_swap 12 \
    --fp8_scaled \
    --fp8_fast \
    --output_dir outputs/longcat_lora
```

Key options:

- `--blocks_to_swap`: Offloads transformer blocks to CPU (works with `accelerate` wrapping).
- `--fp8_scaled` / `--fp8_fast`: Enable FP8 weight optimization and accelerated matmuls.
- `--longcat_i2v`: Switch dataset interpretation to I2V (image-conditioned) mode.
- LoRA modules are registered automatically; ensure `--network_module networks.lora_wan` is set via config or CLI.

<details>
<summary>日本語</summary>

LongCat の LoRA 学習は `longcat_train_network.py` で実施します。WAN トレーナーの拡張版で、LongCat 固有の既定値が設定されています。

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/longcat_train_network.py \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dataset_config path/to/dataset.toml \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --blocks_to_swap 12 \
    --fp8_scaled \
    --fp8_fast \
    --output_dir outputs/longcat_lora
```

主なオプション:

- `--blocks_to_swap`: Transformer ブロックを CPU に退避します（`accelerate` ラッピング後に有効化）。
- `--fp8_scaled` / `--fp8_fast`: FP8 重み最適化・高速行列演算を有効化。
- `--longcat_i2v`: データセットを I2V（画像条件付き）モードとして扱います。
- LoRA モジュールは自動登録されます。`--network_module networks.lora_wan` を設定ファイルまたは CLI で指定してください。

</details>

## Inference / 推論

Use `longcat_generate_video.py` for inference. The script mirrors WAN/Hunyuan options and exposes LongCat-specific modes.

### Text-to-Video (T2V)

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode t2v \
    --prompt "A cat stretching endlessly across the skyline" \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --video_size 480 832 \
    --num_frames 93 \
    --infer_steps 40 \
    --guidance_scale 4.0 \
    --blocks_to_swap 12 \
    --fp8_scaled \
    --save_path outputs/longcat_t2v.mp4
```

### Image-to-Video (I2V)

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode i2v \
    --prompt "The tapestry comes to life" \
    --image_path inputs/frame0.png \
    --i2v_resolution 480p \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --blocks_to_swap 16 \
    --save_path outputs/longcat_i2v.mp4
```

### Refinement

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode refine \
    --stage1_frames stage1/video.mp4 \
    --refine_cond_frames 12 \
    --refine_t_thresh 0.4 \
    --refine_spatial_only \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --save_path outputs/longcat_refine.mp4
```

Important flags:

- `--blocks_to_swap`, `--block_swap_pinned_memory`: identical to WAN/Hunyuan offloading controls.
- `--fp8_scaled`, `--fp8_fast`, `--fp8_t5`: enable FP8 paths for DiT and T5.
- `--attn_mode`: choose between `torch`, `flash`, `flash3`, `xformers`.
- `--lora path[:multiplier]`: load and activate LoRAs trained with LongCat/WAN utilities.

<details>
<summary>日本語</summary>

推論は `longcat_generate_video.py` を使用します。WAN / Hunyuan と同じオプションに加え、LongCat 固有のモードが利用できます。

### テキスト→動画 (T2V)

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode t2v \
    --prompt "A cat stretching endlessly across the skyline" \
    --tokenizer google/umt5-xxl \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --video_size 480 832 \
    --num_frames 93 \
    --infer_steps 40 \
    --guidance_scale 4.0 \
    --blocks_to_swap 12 \
    --fp8_scaled \
    --save_path outputs/longcat_t2v.mp4
```

### 画像→動画 (I2V)

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode i2v \
    --prompt "The tapestry comes to life" \
    --image_path inputs/frame0.png \
    --i2v_resolution 480p \
    --tokenizer google/umt5-xxl \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --blocks_to_swap 16 \
    --save_path outputs/longcat_i2v.mp4
```

### リファイン

```bash
python src/musubi_tuner/longcat_generate_video.py \
    --mode refine \
    --stage1_frames stage1/video.mp4 \
    --refine_cond_frames 12 \
    --refine_t_thresh 0.4 \
    --refine_spatial_only \
    --tokenizer google/umt5-xxl \
    --text_encoder ckpts/longcat/models_t5_umt5-xxl-enc-bf16.pth \
    --dit ckpts/longcat/longcat_dit.safetensors \
    --vae ckpts/longcat/Wan2.1_VAE.pth \
    --save_path outputs/longcat_refine.mp4
```

主なフラグ:

- `--blocks_to_swap`, `--block_swap_pinned_memory`: WAN/Hunyuan と同様のオフロード制御。
- `--fp8_scaled`, `--fp8_fast`, `--fp8_t5`: DiT と T5 の FP8 経路を有効化。
- `--attn_mode`: `torch`, `flash`, `flash3`, `xformers` から選択。
- `--lora path[:multiplier]`: LongCat/WAN 用に学習した LoRA を適用します。

</details>

## Tips / 補足

- LongCat LoRA weights are compatible with WAN loaders because the underlying modules share naming schemes.
- The pipeline converts Wan VAE outputs back to FP32 for video export; expect additional memory usage during decoding.
- The refinement mode expects stage-1 latent/video outputs produced by LongCat or WAN-compatible pipelines.

<details>
<summary>日本語</summary>

- LongCat の LoRA 重みはモジュール命名が共通のため、WAN のローダーでも扱えます。
- パイプラインは Wan VAE の出力を動画保存時に FP32 へ戻すため、デコード時のメモリ使用量に注意してください。
- リファインモードは LongCat または WAN 互換パイプラインで生成した Stage 1 の latent/動画出力を想定しています。

</details>
