# Qwen-Image

## Overview / 概要

This document describes the usage of the Qwen-Image and Qwen-Image-Edit/Edit-2509/Edit-2511/Layered architecture within the Musubi Tuner framework. Qwen-Image is a text-to-image generation model that supports standard text-to-image generation, and Qwen-Image-Edit is a model that supports image editing with control images, Layered is a model that supports image layer segmentation.

Qwen-Image-Edit-2509/2511 can use multiple control images simultaneously. While the official version supports up to 3 images, Musubi Tuner allows specifying any number of images (though correct operation is confirmed only up to 3). Additionally, the sizes of the control images can differ (both during training and inference).

This feature is experimental.

Latent pre-caching, training, and inference options can be found in the `--help` output. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのQwen-Image、Qwen-Image-Edit/Edit-2509/Edit-2511/Layeredアーキテクチャの使用法について説明しています。Qwen-Imageは標準的なテキストから画像生成モデルで、Qwen-Image-Editは制御画像を使った画像編集をサポートするモデル、Layeredは画像のレイヤー分割をサポートするモデルです。

Qwen-Image-Edit-2509/2511は、複数枚の制御画像を同時に使用できます。公式では3枚までですが、Musubi Tunerでは任意の枚数を指定できます（正しく動作するのは3枚までです）。またそれぞれの制御画像のサイズは異なっていても問題ありません（学習時、推論時とも）。

この機能は実験的なものです。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, and Text Encoder (Qwen2.5-VL) models.

Official weights from [Qwen's official weights](https://huggingface.co/Qwen) can be used for DiT, Text Encoder, and VAE respectively. If you want to use the weights for ComfyUI, please follow below.

- **Qwen-Image DiT, Text Encoder (Qwen2.5-VL)**: For Qwen-Image DiT and Text Encoder, download `split_files/diffusion_models/qwen_image_bf16.safetensors` and `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI, respectively. **The fp8_scaled version cannot be used.**

- **VAE**: For VAE, download `split_files/vae/qwen_image_vae.safetensors` similarly from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI. ComfyUI's VAE weights are also now usable.

- **Qwen-Image-Edit DiT**: For Qwen-Image-Edit DiT, download `split_files/diffusion_models/qwen_image_edit_bf16.safetensors`, or for Edit-2509, download `split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors`, and for Edit-2511, download similar from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI. **fp8_e4m3fn cannot be used.** Text Encoder and VAE are same as Qwen-Image.

- **Qwen-Image-Layered VAE**: For Qwen-Image-Layered VAE, download `split_files/vae/qwen_image_layered_vae.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI.

- **Qwen-Image-Layered DiT**: For Qwen-Image-Layered DiT, download `split_files/diffusion_models/qwen_image_layered_bf16.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI. **fp8mixed cannot be used.** Text Encoder is same as Qwen-Image.

Thanks to Comfy-Org for releasing these weights.

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen2.5-VL) のモデルをダウンロードする必要があります。

DiT、Text Encoder、VAEのそれぞれに、[Qwenの公式の重み](https://huggingface.co/Qwen)を使用可能です。ComfyUI用の重みを使用する場合は、以下の通りです。
- **DiT, Text Encoder (Qwen2.5-VL)**: DiTおよびText Encoderは、`split_files/diffusion_models/qwen_image_bf16.safetensors` と `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` を https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI からそれぞれダウンロードしてください。**fp8_scaledバージョンは使用できません。**

- **VAE**: VAEは `split_files/vae/qwen_image_vae.safetensors` を同様に https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI からダウンロードしてください。ComfyUIのVAEの重みも使用できるようになりました。

- **Qwen-Image-Edit DiT**: Qwen-Image-Edit DiTは、`split_files/diffusion_models/qwen_image_edit_bf16.safetensors` を、Edit-2509の場合は `split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors` を https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI からダウンロードしてください。**`fp8_e4m3fn`は使用できません。**Text EncoderとVAEはQwen-Imageと同じです。

これらの重みを公開してくださったComfy-Orgに感謝します。

</details>

### Summary of files to download / ダウンロードするファイルのまとめ

**fp8_scaled and fp8_e4m3fn versions cannot be used.**

**Download from https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI :**
|type|model|file|
|----|--------|--------------|
|DiT|Qwen-Image (no edit)|`split_files/diffusion_models/qwen_image_bf16.safetensors`|
|Text Encoder|Qwen2.5-VL|`split_files/text_encoders/qwen_2.5_vl_7b.safetensors`|
|VAE|Qwen-Image VAE|`split_files/vae/qwen_image_vae.safetensors`|

**Download from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI :**
|type|model|file|
|----|--------|--------------|
|DiT|Qwen-Image-Edit|`split_files/diffusion_models/qwen_image_edit_bf16.safetensors`|
|DiT|Qwen-Image-Edit-2509|`split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors`|
|DiT|Qwen-Image-Edit-2511|`split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors`|

**Download from https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI :**
|type|model|file|
|----|--------|--------------|
|VAE|Qwen-Image-Layered VAE|`split_files/vae/qwen_image_layered_vae.safetensors`|
|DiT|Qwen-Image-Layered|`split_files/diffusion_models/qwen_image_layered_bf16.safetensors`|


## Specifying Model Version / モデルバージョンの指定

When specifying the model version in various scripts, use the following options:
|type|option|note|
|----|--------|----|
|Qwen-Image|`--model_version original`|default, can be omitted|
|Qwen-Image-Edit|`--model_version edit`| |
|Qwen-Image-Edit-2509|`--model_version edit-2509`| |
|Qwen-Image-Edit-2511|`--model_version edit-2511`| |
|Qwen-Image-Layered|`--model_version layered`| |

Note that the `--edit` (for Qwen-Image-Edit) and `--edit_plus` (for Qwen-Image-Edit-2509) flags are also available for backward compatibility.

<details>
<summary>日本語</summary>

様々なスクリプトでモデルバージョンを指定する際には、英語版の表を参考にしてください。

`--edit`（Qwen-Image-Edit）および`--edit_plus`（Qwen-Image-Edit-2509）フラグも後方互換性のために利用可能です。

</details>

## Pre-caching / 事前キャッシング

If you are using Qwen-Image-Edit or Edit-2509/2511, please also refer to the [Qwen-Image-Edit section](./dataset_config.md#qwen-image-edit-and-qwen-image-edit-2509) of the dataset config documentation.

If you are using Qwen-Image-Layered, note the following: Since the Qwen-Image-Layered dataset contains multiple target images, please specify `multiple_target=true` in the dataset config. For details, refer to the [dataset config document](./dataset_config.md#sample-for-image-dataset-with-caption-text-files).


### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Qwen-Image.

```bash
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model \
    --model_version original
```

- Uses `qwen_image_cache_latents.py`.
- The `--vae` argument is required.
- Use the `--model_version` option for Qwen-Image-Edit/Layered training.
- For Qwen-Image-Edit training, control images specified in the dataset config will also be cached as latents.
- For Qwen-Image-Layered training, multiple target images will be cached as latents

<details>
<summary>日本語</summary>

Qwen-Image-EditまたはEdit-2509/2511を使用する場合は、事前にデータセット設定のドキュメントの[Qwen-Image-Editのセクション](./dataset_config.md#qwen-image-edit-and-qwen-image-edit-2509) も参照してください。

Qwen-Image-Layeredを使用する場合は、以下に注意してください。Qwen-Image-Layeredのデータセットには複数枚のターゲット画像が含まれるため、データセット設定で`multiple_target=true`を指定してください。詳細は[データセット設定ドキュメント](./dataset_config.md#sample-for-image-dataset-with-caption-text-files)を参照してください。

latentの事前キャッシングはQwen-Image専用のスクリプトを使用します。

- `qwen_image_cache_latents.py`を使用します。
- `--vae`引数を指定してください。
- Qwen-Image-Editの学習には`--model_version`オプションを適切に指定してください。
- Qwen-Image-Editの学習では、データセット設定で指定されたコントロール画像もlatentsとしてキャッシュされます
- Layeredの学習では、複数のターゲット画像がlatentsとしてキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 1 \
    --model_version original
```

- Uses `qwen_image_cache_text_encoder_outputs.py`.
- Requires the `--text_encoder` (Qwen2.5-VL) argument.
- Use the `--fp8_vl` option to run the Text Encoder in fp8 mode for VRAM savings for <16GB GPUs.
- Specify `--model_version` for Qwen-Image-Edit training. Prompts will be processed with control images to generate appropriate embeddings.

**Technical details on the difference between `--model_version edit` and `--model_version edit-2509` and `--model_version edit-2511`**

Qwen-Image-Edit-2509 and 2511 can use multiple images as control images, so the prompts for obtaining Text Encoder outputs differ from Edit.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `qwen_image_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder` (Qwen2.5-VL) 引数が必要です。
- VRAMを節約するために、fp8 でテキストエンコーダを実行する`--fp8_vl`オプションが使用可能です。VRAMが16GB未満のGPU向けです。
- Qwen-Image-Editの学習には`--model_version`を指定してください。プロンプトがコントロール画像と一緒に処理され、適切な埋め込みが生成されます。

**`--model_version edit`と`--model_version edit-2509`および`--model_version edit-2511`の違いに関する技術的詳細**

Qwen-Image-Edit-2509および2511では複数枚の画像をコントロール画像として使用できるため、Text Encoder出力の取得のためのプロンプトがEditとは異なります。

</details>

## LoRA Training / LoRA学習

Training uses a dedicated script `qwen_image_train_network.py`.

**Standard Qwen-Image Training:**

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --model_version original \
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

For training the image editing model, add the `--model_version` option for Qwen-Image-Edit, Edit-2509, or Edit-2511.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/edit_dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --model_version edit-2511 \
    ...
```

**Qwen-Image-Layered Training:**

For training Qwen-Image-Layered models with layered control images, add the `--model_version layered` option. 

`--remove_first_image_from_target` option is also available to exclude the first target image from the model input/target during training. The first image among multiple target images inferred by the official model in Qwen-Image-Layered is the original image, and the rest are layer images. By using this option, you can train only on the layer images without inferring the original image. This improves training and inference speed and reduces memory usage. The impact on quality is unknown.

Note that VAE is different for this architecture. Please use the VAE model for Qwen-Image-Layered.

For sample image generation during Qwen-Image-Layered training, please refer to [this document](./sampling_during_training.md#sample-image-generation-during-qwen-image-layered-training--qwen-image-layeredの学習中のサンプルイメージ生成).

---

Common notes for Qwen-Image/Qwen-Image-Edit/Layered training:

- Uses `qwen_image_train_network.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- `--mixed_precision bf16` is recommended for Qwen-Image training.
- Use the `--model_version` option for Qwen-Image-Edit, Edit-2509, or Edit-2511 training with control images, or for Qwen-Image-Layered training with multiple target images.
- Memory saving options like `--fp8_base` and `--fp8_scaled` (for DiT), and `--fp8_vl` (for Text Encoder) are available. 
-  `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- `--disable_numpy_memmap`: Disables numpy memory mapping for model loading, loading with standard file read. Increases RAM usage but significantly speeds up model loading in some cases.

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

**Note:** The `--disable_numpy_memmap` option speeds up model loading in some cases with using standard file read instead of using numpy memory mapping. If you encounter slow model weight loading time, this option may help.

<details>
<summary>日本語</summary>

Qwen-Imageの学習は専用のスクリプト`qwen_image_train_network.py`を使用します。コマンドライン例は英語版を参照してください。

**Qwen-Image-Editの学習について**

画像編集モデルの学習には、Qwen-Image-Edit、Edit-2509、またはEdit-2511の`--model_version`オプションを追加してください。

**Qwen-Image-Layeredの学習について**

レイヤード制御画像を使用したQwen-Image-Layeredモデルの学習には、`--model_version layered`オプションを追加してください。

`--remove_first_image_from_target`オプションも利用可能で、学習中に最初のターゲット画像をモデルの入力/ターゲットから除外します。Qwen-Image-Layeredでは公式モデルでは推論される複数枚の画像のうち、最初の画像は元の画像であり、残りがレイヤー画像となっています。このオプションを使用すると、元の画像を推論せずにレイヤー画像のみを学習できます。これにより学習、推論の速度が向上し、メモリ使用量も削減されます。品質への影響は不明です。

このアーキテクチャではVAEが異なることに注意してください。Qwen-Image-Layered用のVAEモデルを使用してください。

Qwen-Image-Layeredにおける学習中のサンプル画像生成については、[こちらのドキュメント](./sampling_during_training.md#sample-image-generation-during-qwen-image-layered-training--qwen-image-layeredの学習中のサンプルイメージ生成)を参照してください。

---

Qwen-Image/Edit/Layered学習に共通の注意点:

- `qwen_image_train_network.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- Qwen-Imageの学習には`--mixed_precision bf16`を推奨します。
- コントロール画像を使ったQwen-Image-Edit/Edit-2509/Edit-2511の学習、複数ターゲット画像を使ったQwen-Image-Layeredの学習には、`--model_version`オプションを適切に指定してください。
- `--fp8_base`や`--fp8_scaled`（DiT用）、`--fp8_vl`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。
- `--disable_numpy_memmap`: モデル読み込み時のnumpyメモリマッピングを無効化し、標準のファイル読み込みで読み込みを行います。RAM使用量は増加しますが、場合によってはモデルの読み込みが大幅に高速化されます。もしモデルの重みの読み込み時間が遅い場合は、このオプションが役立つかもしれません。

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

**備考:** `--disable_numpy_memmap`オプションは、numpyメモリマッピングの代わりに標準のファイル読み込みを使用することで、場合によってはモデルの読み込みを高速化します。モデルの重みの読み込み時間が遅い場合は、このオプションが役立つかもしれません。

</details>

## Finetuning

Finetuning uses a dedicated script `qwen_image_train.py`. This script performs full finetuning of the model, not LoRA. Sample usage is as follows:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/qwen_image_train.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --model_version original \
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
- `--disable_numpy_memmap`: Disables numpy memory mapping for model loading, loading with standard file read. Increases RAM usage but may speed up model loading in some cases.

`--full_bf16` reduces VRAM usage by about 20GB but may impact model accuracy as the weights are kept in bfloat16. Note that the optimizer state is still kept in float32. In addition, it is recommended to use this with an optimizer that supports stochastic rounding. In this repository, Adafactor optimizer with `--fused_backward_pass` option supports stochastic rounding.

When using `--mem_eff_save`, please note that traditional saving methods are still used when saving the optimizer state in `--save_state`, requiring about 40GB of main memory.

`--model_version` option allows for finetuning of Qwen-Image-Edit/Edit-2509/Edit-2511 (unverified).

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
- `--disable_numpy_memmap`: モデル読み込み時のnumpyメモリマッピングを無効化し、標準のファイル読み込みで読み込みを行います。RAM使用量は増加しますが、場合によってはモデルの読み込みが高速化されます。

`--full_bf16`はVRAM使用量を約20GB削減しますが、重みがbfloat16で保持されるため、モデルの精度に影響を与える可能性があります。オプティマイザの状態はfloat32で保持されます。また、効率的な学習のために、stochastic roundingをサポートするオプティマイザとの併用が推奨されます。このリポジトリでは、`adafactor`オプティマイザに`--fused_backward_pass`オプションの組み合わせでstochastic roundingをサポートしています。

`--mem_eff_save`を使用する場合でも、`--save_state`においてはオプティマイザの状態を保存する際に従来の保存方法が依然として使用されるため、約40GBのメインメモリが必要であることに注意してください。

`--model_version`オプションにより、Qwen-Image-Edit/Edit-2509/Edit-2511のfinetuningが可能です（未検証）。

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

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit path/to/edit_dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --model_version edit-2511 \
    --control_image_path path/to/control_image.png \
    --prompt "Change the background to a beach" \
    --resize_control_to_official_size \
    ...
```

**Qwen-Image-Layered Inference:**

Please specify `--model_version layered` for Qwen-Image-Layered inference. Note that VAE is different for this architecture. Please use the VAE model for Qwen-Image-Layered.

---

- Uses `qwen_image_generate_image.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--guidance_scale` controls the classifier-free guidance scale.
- For Qwen-Image-Edit:
    - Use the `--model_version` option to specify the version for image editing mode. For example, `--model_version edit-2511` or `--model_version layered`.
    - `--control_image_path`: Path to the control (reference) image for editing. Edit-2509 also supports multiple arguments (e.g., `--control_image_path img1.png img2.png img3.png`).
    - `--resize_control_to_image_size`: Resize control image to match the specified image size. 
    - `--resize_control_to_official_size`: Resize control image to official size (1M pixels keeping aspect ratio). **Recommended for better results with Edit models.** (Mandatory for 2511)
    - Above two options are mutually exclusive. If both are not specified, the control image will be used at its original resolution.
    - `--append_original_name`: When saving edited images, appends the original base name of the control image to the output file name.
- For Qwen-Image-Layered:
    - Specify the image to be layered in `--control_image_path`.
    - Specify the number of layers to output in `--output_layers`. (Since Qwen-Image-Layered also generates the original image, it generates one more than the specified number. If `--remove_first_image_from_target` was used during training, specify "the number of layers - 1" here to match the number of generated images.)
    - `--resize_control_to_image_size`: Resize control image to match the specified image size. **Recommended for better results with Layered models.**
- Memory saving options like `--fp8_scaled` (for DiT) are available.
- `--text_encoder_cpu` enables CPU inference for the text encoder. Recommended for systems with limited GPU resources (less than 16GB VRAM).
- LoRA loading options (`--lora_weight`, `--lora_multiplier`) are available.

You can specify the discrete flow shift using `--flow_shift`. If omitted, the default value (dynamic shifting based on the image size) will be used.

`xformers`, `flash` and `sageattn` are also available as attention modes. However `sageattn` is not confirmed to work yet.

<details>
<summary>日本語</summary>

Qwen-Imageの推論は専用のスクリプト`qwen_image_generate_image.py`を使用します。コマンド例は英語版のドキュメントを参照してください。

**Qwen-Image-Layeredの推論について**

Qwen-Image-Layeredの推論には`--model_version layered`を指定してください。このアーキテクチャではVAEが異なることに注意してください。Qwen-Image-Layered用のVAEモデルを使用してください。

---

- `qwen_image_generate_image.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- `--guidance_scale`は、classifier-freeガイダンスのスケールを制御します。
- Qwen-Image-Editの場合：
    - 画像編集モードを有効にするために`--model_version`オプションを適切に指定してください。
    - `--control_image_path`: 編集用のコントロール（参照）画像へのパスです。 Edit-2509では複数の引数もサポートしています（例: `--control_image_path img1.png img2.png img3.png`）。
    - `--resize_control_to_image_size`: コントロール画像を指定した画像サイズに合わせてリサイズします。
    - `--resize_control_to_official_size`: コントロール画像を公式サイズ（アスペクト比を保ちながら100万ピクセル）にリサイズします。指定を推奨します（特に2511では必須）。
    - 上記2つのオプションは同時に指定できません。両方とも指定しない場合、制御画像はそのままの解像度で使用されます。
    - `--append_original_name`: 編集された画像を保存する際に、コントロール画像の元の基本名を出力ファイル名に追加します。
- Qwen-Image-Layeredの場合：
    - `--control_image_path`に、分割対象の画像を指定してください。
    - `--output_layers`に出力するレイヤー数を指定してください。（Qwen-Image-Layeredは元画像も生成するため、指定した数より1枚多く生成されます。もし学習時に`--remove_first_image_from_target`を使用していた場合は、ここには「レイヤー数－1」を指定してください。）
    - `--resize_control_to_image_size`: コントロール画像を指定した画像サイズに合わせてリサイズします。Layeredモデルでより良い結果を得るために推奨されます。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。
- `--text_encoder_cpu`を指定するとテキストエンコーダーをCPUで推論します。GPUのVRAMが16GB未満のシステムでは、CPU推論を推奨します。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`）が利用可能です。

`--flow_shift`を指定することで、離散フローシフトを設定できます。省略すると、デフォルト値（画像サイズに基づく動的シフト）が使用されます。

`xformers`、`flash`、`sageattn`もattentionモードとして利用可能です。ただし、`sageattn`はまだ動作確認が取れていません。

</details>

### Inpainting and Reference Consistency Mask (RCM)

For Qwen-Image-Edit, inpainting with a mask image and a feature called Reference Consistency Mask (RCM) are available to prevent unintended changes in the background or other areas.

**These features are only available in Edit/Edit-plus mode, and require the first control image to be the same size as the output image.** They cannot be used at the same time.

- `--mask_path`: Specifies the path to a mask image for inpainting. The image should be black and white, where white areas indicate the regions to be inpainted (changed) and black areas indicate the regions to be preserved.
- `--rcm_threshold`: Enables the Reference Consistency Mask (RCM) feature. RCM is a technique that dynamically creates a mask during the denoising process to prevent unintended modifications to areas that should remain unchanged. It compares the latents of the current generation step with the latents of the control image and protects areas with small differences. Lower values for the threshold result in a larger inpainting area. Typical values are 0.01 to 0.1 for absolute threshold, 0.1 to 0.5 for relative threshold.
- `--rcm_relative_threshold`: If this flag is set, the `--rcm_threshold` is treated as a relative value (0.0-1.0) to the maximum difference observed in the current step. This can provide more stable results across different steps. If not set, the threshold is an absolute value.
- `--rcm_kernel_size`: Specifies the kernel size for a Gaussian blur applied before calculating the difference. This helps to create a smoother, more stable mask. Default is 3.
- `--rcm_dilate_size`: Specifies the size to dilate (expand) the inpainting region of the generated mask. This is useful for ensuring that the edges of the area you want to change are properly modified. Default is 0 (no dilation).
- `--rcm_debug_save`: When this flag is set, the dynamically generated RCM mask for each step will be saved in the output directory. This is very useful for debugging and adjusting the RCM parameters.

**Example using RCM:**

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit path/to/edit_dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --edit \
    --control_image_path path/to/control_image.png \
    --prompt "Change her dress to red" \
    --image_size 1024 1024 \
    --rcm_threshold 0.2 --rcm_relative_threshold \
    --rcm_kernel_size 3 --rcm_dilate_size 1 \
    ...
```

#### Important Usage Notes

-   **Compatibility:** Both RCM and the standard inpainting mask are only effective in **edit mode** (when a control image is provided).
-   **Requirement:** To use these features, the initial control image must have the **same dimensions** as the final output image. The script will show an error and disable RCM if the sizes do not match.
-   **Exclusivity:** RCM and `--mask_path` cannot be used at the same time.
-   **Debugging Tip:** When first using RCM, it is highly recommended to use the `--rcm_debug_save` flag. This will save the masks to the output directory, allowing you to visually inspect how the `threshold` and other parameters are affecting the mask generation.

#### Technical Details of RCM

Reference Consistency Mask (RCM) addresses a common issue in Qwen-Image-Edit where the generated image has a slight positional drift or misalignment compared to the control image. RCM significantly improves the structural stability and positional accuracy of the image editing process.

This feature is implemented based on the idea of dynamically creating a mask during the denoising loop to "anchor" the parts of the image that should remain consistent with the reference (control) image.

**How RCM Works**

For each step in the denoising loop, RCM performs the following actions:
1.  It calculates a "noisy" version of the original control latent, corresponding to the current timestep `t`.
2.  It computes the difference between the current generation latent and the noisy control latent.
3.  Areas with a small difference are considered "consistent" and are masked to be preserved. The sensitivity is controlled by the `rcm_threshold`.
4.  This mask is then used to reset the consistent regions of the current latent back to the state of the noisy reference latent, just before the `scheduler.step` is called.

This self-correcting mechanism prevents the accumulation of positional errors throughout the denoising process, ensuring that unchanged elements like backgrounds or faces stay perfectly aligned.

<details>
<summary>日本語</summary>

Qwen-Image-Editにおいて、背景などを意図せず変更してしまうことを防ぐため、マスク画像を使ったInpaintingと、Reference Consistency Mask (RCM) という機能が利用可能です。

**これらの機能はEdit/Edit-plusモードでのみ利用可能で、かつ最初のコントロール画像が出力画像と同じサイズである必要があります。** また、同時に使用することはできません。

- `--mask_path`: Inpainting用のマスク画像へのパスを指定します。白黒のマスク画像で、白の領域がInpainting（変更）される領域、黒の領域が維持される領域を示します。
- `--rcm_threshold`: Reference Consistency Mask (RCM) 機能を有効にします。RCMは、Denoisingの過程で動的にマスクを生成し、変更すべきでない箇所が意図せず変更されるのを防ぐ技術です。現在の生成ステップのlatentとコントロール画像のlatentを比較し、差が小さい部分を保護します。閾値が低いほど、Inpainting領域は大きくなります。
- `--rcm_relative_threshold`: このフラグを指定すると、`--rcm_threshold`がそのステップで観測された差分の最大値に対する相対的な値（0.0～1.0）として扱われます。これにより、ステップごとに安定した結果が得られやすくなります。指定しない場合は絶対値として扱われます。絶対値の場合は0.01～0.1、相対値の場合は0.1～0.5が典型的な値です。
- `--rcm_kernel_size`: 差分を計算する前に適用するガウシアンブラーのカーネルサイズを指定します。これにより、より滑らかで安定したマスクが生成されます。デフォルトは3です。
- `--rcm_dilate_size`: 生成されたマスクのInpainting領域を膨張（dilate）させるサイズを指定します。変更したい領域の境界部分が確実に変更されるようにしたい場合に便利です。デフォルトは0（膨張なし）です。
- `--rcm_debug_save`: このフラグを指定すると、各ステップで動的に生成されたRCMのマスクが出力ディレクトリに保存されます。RCMのパラメータを調整する際のデバッグに非常に役立ちます。

**重要な使用上の注意**

-   **互換性:** RCMと標準のinpaintingマスクは、どちらも**Editモード**（制御画像が提供されている場合）でのみ有効です。
-   **要件:** これらの機能を使用するには、最初の制御画像が最終的な出力画像と**同じサイズ**である必要があります。サイズが一致しない場合、スクリプトはエラーを表示し、RCMを無効にします。
-   **排他性:** RCMと`--mask_path`は同時に使用できません。
-   **デバッグのヒント:** 初めてRCMを使用する際は、`--rcm_debug_save`フラグを使用することを強く推奨します。これによりマスクが出力ディレクトリに保存され、`threshold`などのパラメータがマスク生成にどのように影響しているかを視覚的に確認できます。

**RCMの技術的詳細**

Reference Consistency Mask (RCM) は、Qwen-Image-Editにおいて、生成画像が制御画像と比較してわずかな位置ずれを起こすという一般的な問題を解決するためのものです。RCMは、編集プロセスにおける構造的な安定性と位置精度を大幅に向上させます。

この機能は、denoisingループ中に動的にマスクを生成し、参照元（制御画像）と一致すべき部分を「固定（アンカー）」するというアイデアに基づいています。

**RCMの動作原理**

RCMは、denoisingループの各ステップで以下の処理を実行します。
1.  現在のタイムステップ`t`に対応する、元の制御画像の潜在変数にノイズを加えたバージョンを計算します。
2.  現在の生成中latentと、ノイズ付加済み制御latentとの差分を計算します。
3.  差分が小さい領域を「一致している」とみなし、その部分を保持するようにマスクします。この感度は`rcm_threshold`によって制御されます。
4.  そして、このマスクを使い、`scheduler.step`が呼び出される直前に、一致している領域をノイズ付加済み参照latentの状態にリセットします。

この自己修正的なメカニズムにより、denoisingプロセス全体を通して位置誤差が蓄積されるのを防ぎ、背景や顔のような変更しない要素が完全に位置ずれなく維持されることを保証します。

</details>