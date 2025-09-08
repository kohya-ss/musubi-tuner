# Qwen-Image

## Overview / 概要

This document describes the usage of the Qwen-Image and Qwen-Image-Edit architecture within the Musubi Tuner framework. Qwen-Image is a text-to-image generation model that supports standard text-to-image generation, and Qwen-Image-Edit is a model that supports image editing with control images.

This feature is experimental.

Latent pre-caching, training, and inference options can be found in the `--help` output. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのQwen-Image、Qwen-Image-Editアーキテクチャの使用法について説明しています。Qwen-Imageは標準的なテキストから画像生成モデルで、Qwen-Image-Editは制御画像を使った画像編集をサポートするモデルです。

この機能は実験的なものです。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, and Text Encoder (Qwen2.5-VL) models.

- **Qwen-Image DiT, Text Encoder (Qwen2.5-VL)**: For Qwen-Image DiT and Text Encoder, download `split_files/diffusion_models/qwen_image_bf16.safetensors` and `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI, respectively. **The fp8_scaled version cannot be used.**

- **VAE**: For VAE, download `vae/diffusion_pytorch_model.safetensors` from https://huggingface.co/Qwen/Qwen-Image. **ComfyUI's VAE weights cannot be used.**

- **Qwen-Image-Edit DiT**: For Qwen-Image-Edit DiT, download `split_files/diffusion_models/qwen_image_edit_bf16.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI. Text Encoder and VAE are same as Qwen-Image.

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen2.5-VL) のモデルをダウンロードする必要があります。

- **DiT, Text Encoder (Qwen2.5-VL)**: DiTおよびText Encoderは、`split_files/diffusion_models/qwen_image_bf16.safetensors` と `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` を https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI からそれぞれダウンロードしてください。**fp8_scaledバージョンは使用できません。**

- **VAE**: VAEは `vae/diffusion_pytorch_model.safetensors` を https://huggingface.co/Qwen/Qwen-Image からダウンロードしてください。**ComfyUIのVAEウェイトは使用できません。**

- **Qwen-Image-Edit DiT**: Qwen-Image-Edit DiTは、`split_files/diffusion_models/qwen_image_edit_bf16.safetensors` を https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI からダウンロードしてください。Text EncoderとVAEはQwen-Imageと同じです。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Qwen-Image.

```bash
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
```

- Uses `qwen_image_cache_latents.py`.
- The `--vae` argument is required.
- For Qwen-Image-Edit training, control images specified in the dataset config will also be cached as latents.

<details>
<summary>日本語</summary>

latentの事前キャッシングはQwen-Image専用のスクリプトを使用します。

- `qwen_image_cache_latents.py`を使用します。
- `--vae`引数を指定してください。
- Qwen-Image-Editの学習では、データセット設定で指定されたコントロール画像もlatentsとしてキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 1
```

- Uses `qwen_image_cache_text_encoder_outputs.py`.
- Requires the `--text_encoder` (Qwen2.5-VL) argument.
- Use the `--fp8_vl` option to run the Text Encoder in fp8 mode for VRAM savings for <16GB GPUs.
- For Qwen-Image-Edit training, add `--edit` flag. Prompts will be processed with control images to generate appropriate embeddings.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `qwen_image_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder` (Qwen2.5-VL) 引数が必要です。
- VRAMを節約するために、fp8 でテキストエンコーダを実行する`--fp8_vl`オプションが使用可能です。VRAMが16GB未満のGPU向けです。
- Qwen-Image-Editの学習では、`--edit`フラグを追加してください。プロンプトがコントロール画像と一緒に処理され、適切な埋め込みが生成されます。

</details>

## LoRA Training / LoRA学習

Training uses a dedicated script `qwen_image_train_network.py`.

**Standard Qwen-Image Training:**

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift \
    --weighting_scheme none --discrete_flow_shift 2.2 \
    --optimizer_type adamw8bit --learning_rate 5e-5 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

**Qwen-Image-Edit Training:**

For training the image editing model, add the `--edit` flag:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/edit_dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --edit \
    ...
```

- Uses `qwen_image_train_network.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- `--mixed_precision bf16` is recommended for Qwen-Image training.
- Add `--edit` flag for Qwen-Image-Edit training with control images.
- Memory saving options like `--fp8_base` and `--fp8_scaled` (for DiT), and `--fp8_vl` (for Text Encoder) are available. 
-  `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

`--fp8_vl` is recommended for GPUs with less than 16GB of VRAM.

`--sdpa` uses PyTorch's scaled dot product attention. Other options like `--xformers` and `--flash_attn` are available. `flash3` cannot be used currently.

If you specify `--split_attn`, the attention computation will be split, slightly reducing memory usage. Please specify `--split_attn` if you are using anything other than `--sdpa`.

`--timestep_sampling` allows you to choose the sampling method for the timesteps. `shift` with `--discrete_flow_shift` is the default. `qwen_shift` is also available. `qwen_shift` is a same method during inference. It uses the dynamic shift value based on the resolution of each image (typically around 2.2 for 1328x1328 images).

`--discrete_flow_shift` is set quite low for Qwen-Image during inference (as described), so a lower value than other models may be preferable.

Don't forget to specify `--network_module networks.lora_qwen_image`.

The appropriate settings for each parameter are unknown. Feedback is welcome.

### VRAM Usage Estimates with Memory Saving Options

For 1024x1024 training with the batch size of 1, `--mixed_precision bf16` and `--gradient_checkpointing` is enabled and `--xformers` is used.

|options|VRAM Usage|
|-------|----------|
|no   |42GB|
|`--fp8_base --fp8_scaled`|30GB|
|+ `--blocks_to_swap 16`|24GB|
|+ `--blocks_to_swap 45`|12GB|

64GB main RAM system is recommended with `--blocks_to_swap`.

If `--blocks_to_swap` is more than 45, the main RAM usage will increase significantly.

Qwen-Image-Edit training requires additional memory for the control images.

<details>
<summary>日本語</summary>

Qwen-Imageの学習は専用のスクリプト`qwen_image_train_network.py`を使用します。

- `qwen_image_train_network.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- Qwen-Imageの学習には`--mixed_precision bf16`を推奨します。
- コントロール画像を使ったQwen-Image-Editの学習には`--edit`フラグを追加します。
- `--fp8_base`や`--fp8_scaled`（DiT用）、`--fp8_vl`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。

GPUのVRAMが16GB未満の場合は、`--fp8_vl`を推奨します。

`--sdpa`はPyTorchのscaled dot product attentionを用います。他に `--xformers`、`--flash_attn` があります。`--flash3`は現在使用できません。

`--split_attn` を指定すると、attentionの計算が分割され、メモリ使用量がわずかに削減されます。`--sdpa` 以外を使用する場合は、`--split_attn` を指定してください。

`--timestep_sampling` では、タイムステップのサンプリング方法を選択できます。`shift` と `--discrete_flow_shift` の組み合わせがデフォルトです。`qwen_shift` も利用可能です。`qwen_shift` は推論時と同じ方法で、各画像の解像度に基づいた動的シフト値を使用します（通常、1328x1328画像の場合は約2.2です）。

`--discrete_flow_shift`は、Qwen-Imageでは前述のように推論時にかなり低めなため、他のモデルよりも低めが良いかもしれません。

`--network_module networks.lora_qwen_image`を指定することを忘れないでください。

それぞれのパラメータの適切な設定は不明です。フィードバックをお待ちしています。

### メモリ節約オプションを使用した場合のVRAM使用量の目安

1024x1024の学習でバッチサイズ1の場合、`--mixed_precision bf16`と`--gradient_checkpointing`を指定し、`--xformers`を使用した場合のVRAM使用量の目安は以下の通りです。

|オプション|VRAM使用量|
|-------|----------|
|no   |42GB|
|`--fp8_base --fp8_scaled`|30GB|
|+ `--blocks_to_swap 16`|24GB|
|+ `--blocks_to_swap 45`|12GB|

`--blocks_to_swap`を使用する場合は、64GBのメインRAMを推奨します。

`--blocks_to_swap`が45を超えると、メインRAMの使用量が大幅に増加します。

Qwen-Image-Editの学習では、コントロール画像のために追加のメモリが必要です。

</details>

## Finetuning

Finetuning uses a dedicated script `qwen_image_train.py`. This script performs full finetuning of the model, not LoRA. Sample usage is as follows:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/qwen_image_train.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 --gradient_checkpointing \
    --optimizer_type adafactor --learning_rate 1e-6 --fused_backward_pass \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --max_grad_norm 0 --lr_scheduler constant_with_warmup --lr_warmup_steps 10 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-model
```

- Uses `qwen_image_train.py`.
- Finetuning requires a large amount of VRAM. The use of memory saving options is strongly recommended.
- `--full_bf16`: Loads the model weights in bfloat16 format to significantly reduce VRAM usage. 
- `--optimizer_type adafactor`: Using Adafactor is recommended for finetuning.
- `--fused_backward_pass`: Reduces VRAM usage during the backward pass when using Adafactor.
- `--mem_eff_save`: Reduces main memory (RAM) usage when saving checkpoints.
- `--blocks_to_swap`: Swaps model blocks between VRAM and main memory to reduce VRAM usage. This is effective when VRAM is limited.

`--full_bf16` reduces VRAM usage by about 20GB but may impact model accuracy as the weights are kept in bfloat16. Note that the optimizer state is still kept in float32. In addition, it is recommended to use this with an optimizer that supports stochastic rounding. In this repository, Adafactor optimizer with `--fused_backward_pass` option supports stochastic rounding.

For `--mem_eff_save`, please note that when saving the optimizer state with `--save_state`, the conventional saving method is used, which will still require about 40GB of main memory.

`--edit` option allows for finetuning of Qwen-Image-Edit (unverified).

### Recommended Settings

We are still exploring the optimal settings. The configurations above are just examples, so please adjust them as needed. We welcome your feedback.

If you have ample VRAM, you can use any optimizer of your choice. `--full_bf16` is not recommended.

For limited VRAM environments (e.g., 48GB or less), you may need to use `--full_bf16`, the Adafactor optimizer, and `--fused_backward_pass`. Settings above are the recommended options for that case. Please adjust `--lr_warmup_steps` to a value between approximately 10 and 100.

`--fused_backward_pass` is not currently compatible with gradient accumulation, and max grad norm may not function as expected, so it is recommended to specify `--max_grad_norm 0`.

If your VRAM is even more constrained, you can enable block swapping by specifying a value for `--blocks_to_swap`.

Experience with other models suggests that the learning rate may need to be reduced significantly; something in the range of 1e-6 to 1e-5 might be a good place to start.

<details>
<summary>日本語</summary>

Finetuningは専用のスクリプト`qwen_image_train.py`を使用します。このスクリプトはLoRAではなく、モデル全体のfinetuningを行います。

- `qwen_image_train.py`を使用します。
- Finetuningは大量のVRAMを必要とします。メモリ節約オプションの使用を強く推奨します。
- `--full_bf16`: モデルの重みをbfloat16形式で読み込み、VRAM使用量を大幅に削減します。
- `--optimizer_type adafactor`: FinetuningではAdafactorの使用が推奨されます。
- `--fused_backward_pass`: Adafactor使用時に、backward pass中のVRAM使用量を削減します。
- `--mem_eff_save`: チェックポイント保存時のメインメモリ（RAM）使用量を削減します。
- `--blocks_to_swap`: モデルのブロックをVRAMとメインメモリ間でスワップし、VRAM使用量を削減します。VRAMが少ない場合に有効です。

`--full_bf16`はVRAM使用量を約20GB削減しますが、重みがbfloat16で保持されるため、モデルの精度に影響を与える可能性があります。オプティマイザの状態はfloat32で保持されます。また、効率的な学習のために、stochastic roundingをサポートするオプティマイザとの併用が推奨されます。このリポジトリでは、`adafactor`オプティマイザに`--fused_backward_pass`オプションの組み合わせでstochastic roundingをサポートしています。

`--mem_eff_save`を使用する場合、`--save_state`でオプティマイザの状態を保存する際に従来の保存方法が使用されるため、約40GBのメインメモリが依然として必要であることに注意してください。

`--edit`オプションを追加するとQwen-Image-Editのfinetuningが可能です（未検証です）。

### 推奨設定

最適な設定はまだ調査中です。上記の構成はあくまで一例ですので、必要に応じて調整してください。フィードバックをお待ちしております。

十分なVRAMがある場合は、お好みのオプティマイザを使用できます。`--full_bf16`は推奨されません。

VRAMが限られている環境（例：48GB以下）の場合は、`--full_bf16`、Adafactorオプティマイザ、および`--fused_backward_pass`を使用する必要があるかもしれません。上記の設定はその場合の推奨オプションです。`--lr_warmup_steps`は約10から100の間の値に調整してください。

現時点では`--fused_backward_pass`はgradient accumulationに対応していません。またmax grad normも想定通りに動作しない可能性があるため、`--max_grad_norm 0`を指定することを推奨します。

さらにVRAMが制約されている場合は、`--blocks_to_swap`に値を指定してブロックスワッピングを有効にできます。

</details>

## Inference / 推論

Inference uses a dedicated script `qwen_image_generate_image.py`.

**Standard Qwen-Image Inference:**

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --prompt "A cat" \
    --negative_prompt " " \
    --image_size 1024 1024 --infer_steps 25 \
    --guidance_scale 4.0 \
    --attn_mode sdpa \
    --save_path path/to/save/dir \
    --output_type images \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

**Qwen-Image-Edit Inference:**

For image editing with control images, add the `--edit` flag and specify a control image:

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit path/to/edit_dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --edit \
    --control_image_path path/to/control_image.png \
    --prompt "Change the background to a beach" \
    --resize_control_to_official_size \
    ...
```

- Uses `qwen_image_generate_image.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--guidance_scale` controls the classifier-free guidance scale.
- For Qwen-Image-Edit:
  - Add `--edit` flag to enable image editing mode.
  - `--control_image_path`: Path to the control (reference) image for editing.
  - `--resize_control_to_image_size`: Resize control image to match the specified image size.
  - `--resize_control_to_official_size`: Resize control image to official size (1M pixels keeping aspect ratio). Recommended for better results.
  - Above two options are mutually exclusive. If both are not specified, the control image will be used at its original resolution.
- Memory saving options like `--fp8_scaled` (for DiT) are available.
- `--text_encoder_cpu` enables CPU inference for the text encoder. Recommended for systems with limited GPU resources (less than 16GB VRAM).
- LoRA loading options (`--lora_weight`, `--lora_multiplier`) are available.

You can specify the discrete flow shift using `--flow_shift`. If omitted, the default value (dynamic shifting based on the image size) will be used.

`xformers`, `flash` and `sageattn` are also available as attention modes. However `sageattn` is not confirmed to work yet.

<details>
<summary>日本語</summary>

Qwen-Imageの推論は専用のスクリプト`qwen_image_generate_image.py`を使用します。

- `qwen_image_generate_image.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- `--guidance_scale`は、classifier-freeガイダンスのスケールを制御します。
- Qwen-Image-Editの場合:
  - 画像編集モードを有効にするために`--edit`フラグを追加します。
  - `--control_image_path`: 編集用のコントロール（参照）画像へのパスです。
  - `--resize_control_to_image_size`: コントロール画像を指定した画像サイズに合わせてリサイズします。
  - `--resize_control_to_official_size`: コントロール画像を公式サイズ（アスペクト比を保ちながら100万ピクセル）にリサイズします。指定を推奨します。
  - 上記2つのオプションは同時に指定できません。両方とも指定しない場合、制御画像はそのままの解像度で使用されます。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。
- `--text_encoder_cpu`を指定するとテキストエンコーダーをCPUで推論します。GPUのVRAMが16GB未満のシステムでは、CPU推論を推奨します。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`）が利用可能です。

`--flow_shift`を指定することで、離散フローシフトを設定できます。省略すると、デフォルト値（画像サイズに基づく動的シフト）が使用されます。

`xformers`、`flash`、`sageattn`もattentionモードとして利用可能です。ただし、`sageattn`はまだ動作確認が取れていません。

</details>
