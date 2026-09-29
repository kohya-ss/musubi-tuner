# Qwen-Image 2.1

## Overview / 概要

This document describes how to train Qwen-Image 2.1 within the Musubi Tuner framework. Qwen-Image 2.1 supports text-to-image generation and image editing with one or more control images.

Qwen-Image 2.1 uses a dedicated DiT, RGBA VAE, and Qwen3-VL-8B text encoder. Models and caches from the earlier Qwen-Image/Edit architectures cannot be reused. Use the dedicated scripts below; the `--model_version` option is not used.

This feature is experimental.

Options can be found in the `--help` output. Many options are shared with other architectures, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのQwen-Image 2.1の学習方法について説明します。Qwen-Image 2.1は、テキストからの画像生成と、1枚または複数枚の制御画像を使用した画像編集をサポートします。

専用のDiT、RGBA VAE、Qwen3-VL-8Bテキストエンコーダーを使用します。従来のQwen-Image/Editのモデルやキャッシュは再利用できません。以下の専用スクリプトを使用してください。`--model_version`オプションは使用しません。

この機能は実験的なものです。オプションは`--help`で確認してください。必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

</details>

## Download the model / モデルのダウンロード

Prepare the Qwen-Image 2.1 DiT, VAE, and Qwen3-VL-8B text encoder. Pass local paths to the scripts.

- **DiT**: `--dit` accepts a Diffusers transformer directory, a single safetensors file, or the first shard. ComfyUI BF16/FP32 weights with fused `gate_up` projections are also supported. For single-file weights with a non-default model configuration, specify `--dit_config`.
- **VAE**: `--vae` accepts a Diffusers VAE directory or a single safetensors file with Diffusers keys.
- **Text Encoder**: `--text_encoder` accepts a Transformers model directory or a Qwen3-VL-8B-Instruct BF16/FP32 safetensors file.

The processor and tokenizer are loaded automatically from the `processor` subfolder of `Qwen/Qwen-Image-2.1`, using the Hugging Face cache when available.

GGUF, NVFP4, and prequantized fused weights without quantization metadata are not supported.

<details>
<summary>日本語</summary>

Qwen-Image 2.1のDiT、VAE、Qwen3-VL-8Bテキストエンコーダーを用意し、ローカルパスを指定してください。

- **DiT**: `--dit`にはDiffusersのtransformerディレクトリ、単一のsafetensorsファイル、または最初の分割ファイルを指定できます。`gate_up`が結合されたComfyUIのBF16/FP32重みにも対応します。単一ファイルで標準と異なるモデル構成を使用する場合は、`--dit_config`を指定してください。
- **VAE**: `--vae`にはDiffusersのVAEディレクトリ、またはDiffusersのキー形式のsafetensorsファイルを指定します。
- **Text Encoder**: `--text_encoder`にはTransformersのモデルディレクトリ、またはQwen3-VL-8B-InstructのBF16/FP32 safetensorsファイルを指定します。

processorとtokenizerは`Qwen/Qwen-Image-2.1`の`processor`サブフォルダーから自動で読み込まれます。利用可能な場合はHugging Faceのキャッシュを使用します。

GGUF、NVFP4、量子化メタデータのない量子化済み結合重みには対応していません。

</details>

## Pre-caching / 事前キャッシング

Prepare an image dataset using the [dataset configuration documentation](./dataset_config.md).

Each item has one target image and a caption. For editing, specify `control_directory` or the consecutive JSONL fields `control_path_0`, `control_path_1`, etc. Control images are used in the specified order. Bucket dimensions are aligned to multiples of 32.

Use the same dataset configuration for latent and text encoder caching. After changing control images or crop settings, regenerate both caches without `--skip_existing`.

RGB images receive an opaque alpha channel for the VAE. Transparent regions in control images are composited over white for the text encoder.

`--batch_size` controls the encoding batch size. The VAE groups equal-sized images within each batch; the text encoder supports different caption lengths and reference counts. Reduce the batch size if encoding runs out of memory.

<details>
<summary>日本語</summary>

[データセット設定のドキュメント](./dataset_config.md)を参照して画像データセットを用意してください。

各項目には1枚の学習対象画像とキャプションを指定します。編集学習では`control_directory`、またはJSONLの`control_path_0`、`control_path_1`などの連番フィールドを指定してください。制御画像は指定順に使用されます。バケットの解像度は32の倍数に揃えられます。

latentとテキストエンコーダーのキャッシュには同じデータセット設定を使用してください。制御画像やクロップ設定を変更した場合は、`--skip_existing`を使用せずに両方のキャッシュを再生成してください。

RGB画像にはVAE用の不透明なアルファチャンネルが追加されます。テキストエンコーダーでは、制御画像の透明部分を白背景に合成します。

`--batch_size`でエンコードのバッチサイズを指定します。VAEはバッチ内の同じサイズの画像をまとめて処理します。テキストエンコーダーは異なるキャプション長や制御画像数に対応します。メモリが不足する場合はバッチサイズを減らしてください。

</details>

### Latent Pre-caching / latentの事前キャッシング

```bash
python src/musubi_tuner/qwen_image_21_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model \
    --vae_tiling
```

- The `--vae` argument is required.
- Control images specified in the dataset configuration are also cached.
- Use `--vae_tiling` to reduce VRAM usage when encoding large images.

<details>
<summary>日本語</summary>

- `--vae`引数を指定してください。
- データセット設定で指定した制御画像もキャッシュされます。
- 大きな画像のエンコード時にVRAM使用量を減らすには、`--vae_tiling`を指定してください。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

```bash
python src/musubi_tuner/qwen_image_21_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 1
```

- The `--text_encoder` argument is required. The processor is loaded automatically.
- Captions are processed together with control images for editing datasets.
- Use `--fp8_vl` to reduce text encoder VRAM usage. The default dtype is BF16.
- For CPU caching, specify `--device cpu --text_encoder_dtype float32`.

<details>
<summary>日本語</summary>

- `--text_encoder`を指定してください。processorは自動で読み込まれます。
- 編集用データセットでは、キャプションと制御画像を一緒に処理します。
- VRAM使用量を減らすには`--fp8_vl`を指定してください。既定のdtypeはBF16です。
- CPUでキャッシュを作成する場合は、`--device cpu --text_encoder_dtype float32`を指定してください。

</details>

## LoRA Training / LoRA学習

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    src/musubi_tuner/qwen_image_21_train_network.py \
    --dit path/to/dit_model \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 --gradient_checkpointing \
    --network_module networks.lora_qwen_image_21 \
    --network_dim 16 --network_alpha 16 \
    --optimizer_type AdamW --learning_rate 1e-4 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --weighting_scheme none \
    --max_train_steps 2000 --save_every_n_steps 200 \
    --output_dir path/to/output --output_name qwen_image_21_lora
```

- Uses `qwen_image_21_train_network.py`. Specify `networks.lora_qwen_image_21` for `--network_module`.
- Adjust the learning rate, network dimensions, and number of steps for your dataset.
- The VAE and text encoder are not required during training unless generating sample images.
- Use `--blocks_to_swap` to reduce VRAM usage. The number must be less than 32 for the default model. See the [block swap documentation](./block_swap.md).
- `--fp8_base` and `--fp8_scaled` can reduce DiT memory usage. Specify both options for scaled FP8.
- `--sdpa` uses PyTorch scaled dot product attention. SageAttention is not supported for training.
- Attention is always split by sample and text/image segment. `--split_attn` is enabled by default and does not need to be specified.
- Resolution-dependent timestep sampling, such as `qwen_shift`, uses the number of target latent tokens (`H * W`).

<details>
<summary>日本語</summary>

- `qwen_image_21_train_network.py`を使用します。`--network_module`には`networks.lora_qwen_image_21`を指定してください。
- 学習率、ネットワークの次元数、学習ステップ数はデータセットに応じて調整してください。
- サンプル画像を生成しない場合、学習時にVAEとテキストエンコーダーは不要です。
- VRAM使用量を減らすには`--blocks_to_swap`を指定してください。標準モデルでは32未満の値を指定します。[ブロックスワップのドキュメント](./block_swap.md)も参照してください。
- `--fp8_base`と`--fp8_scaled`でDiTのメモリ使用量を減らせます。scaled FP8を使用する場合は両方を指定してください。
- `--sdpa`はPyTorchのscaled dot product attentionを使用します。SageAttentionは学習には対応していません。
- Attentionは常にサンプルおよびテキスト・画像の区間ごとに分割されます。`--split_attn`は既定で有効なため、指定する必要はありません。
- `qwen_shift`などの解像度に依存するタイムステップサンプリングでは、学習対象画像のlatentトークン数（`H * W`）を使用します。

</details>

## Sampling during training / 学習中のサンプル画像生成

To generate sample images, add `--sample_prompts path/to/prompts.txt` and `--sample_every_n_steps 200` to the training command. Also specify `--vae` and `--text_encoder`.

See [Sampling during training](./sampling_during_training.md) for the prompt format. Use repeated `--ci` options for multiple control images:

```text
A cat holding a sign --w 1024 --h 1024 --s 40 --d 42 --ci path/to/control0.png --ci path/to/control1.png
```

`--l` sets the CFG scale (default: 1.0, no CFG). To enable CFG, specify a value greater than 1 and a negative prompt with `--n`.

Sampling uses resolution-dependent dynamic shifting by default. `--fs` overrides it with a fixed flow shift. This is separate from `--discrete_flow_shift` in the training command. Sample PNG images retain the VAE's alpha channel. Repeated prompts with the same ordered control images share text encoder outputs.

<details>
<summary>日本語</summary>

サンプル画像を生成するには、学習コマンドに`--sample_prompts path/to/prompts.txt`と`--sample_every_n_steps 200`を追加してください。`--vae`と`--text_encoder`も指定してください。

プロンプトの形式は[学習中のサンプル画像生成](./sampling_during_training.md)を参照してください。複数の制御画像を使用する場合は、上記の例のように`--ci`を繰り返して指定します。

`--l`でCFGスケールを指定します（既定値: 1.0、CFGなし）。CFGを有効にするには、1より大きい値と`--n`によるネガティブプロンプトを指定してください。

サンプリングでは既定で解像度に応じた動的シフトを使用します。`--fs`を指定すると固定のflow shiftで上書きします。学習コマンドの`--discrete_flow_shift`とは別の設定です。サンプルPNGはVAEのアルファチャンネルを保持します。同じプロンプトと同じ順序の制御画像にはテキストエンコーダーの出力を再利用します。

</details>

## Inference / 推論

Use `qwen_image_21_generate_image.py` to generate images with a trained LoRA:

```bash
python src/musubi_tuner/qwen_image_21_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --prompt "A cat holding a sign" \
    --image_size 1024 1024 --infer_steps 40 --seed 42 \
    --lora_weight path/to/lora.safetensors --lora_multiplier 1.0 \
    --attn_mode sdpa --save_path path/to/output
```

- `--image_size` specifies height and width. Both must be multiples of 32.
- For editing, add `--control_image_path path/to/control0.png path/to/control1.png`. Control images retain their order and are resized to approximately 1M pixels while preserving their aspect ratio.
- Omit `--lora_weight` to use the base model. Multiple LoRAs can be supplied with matching multipliers; a single multiplier applies to all supplied LoRAs.
- Use `--include_patterns` and `--exclude_patterns` to select LoRA modules by regular expression, with one pattern per weight file. For LyCORIS adapters, install `lycoris-lora` and add `--lycoris`. Adapters are merged before FP8 quantization.
- `--guidance_scale` defaults to 1.0. Values greater than 1 require `--negative_prompt`.
- `--flow_shift` specifies a fixed flow shift. If omitted, sampling uses dynamic shifting, as in training previews.
- `--text_encoder_cpu`, `--fp8_vl`, `--fp8_scaled`, `--blocks_to_swap`, and `--vae_tiling` are available to reduce VRAM usage.
- Output PNGs retain the alpha channel.

<details>
<summary>日本語</summary>

学習したLoRAで画像を生成するには`qwen_image_21_generate_image.py`を使用します。

- `--image_size`には高さ、幅の順に指定します。どちらも32の倍数である必要があります。
- 編集には`--control_image_path path/to/control0.png path/to/control1.png`を追加します。制御画像は指定順を保持し、アスペクト比を維持して約100万画素にリサイズされます。
- ベースモデルを使用する場合は`--lora_weight`を省略します。複数のLoRAと対応する倍率を指定できます。倍率が1個の場合はすべてのLoRAに適用されます。
- `--include_patterns`と`--exclude_patterns`でLoRAモジュールを正規表現で選択できます。重みファイルごとに1つのパターンを指定します。LyCORISを使用する場合は`lycoris-lora`をインストールし、`--lycoris`を指定してください。アダプターはFP8量子化の前にマージされます。
- `--guidance_scale`の既定値は1.0です。1より大きい値には`--negative_prompt`が必要です。
- `--flow_shift`で固定シフトを指定します。省略時は学習中のプレビューと同じ動的シフトを使用します。
- VRAM使用量を減らすには`--text_encoder_cpu`、`--fp8_vl`、`--fp8_scaled`、`--blocks_to_swap`、`--vae_tiling`が利用できます。
- 出力PNGはアルファチャンネルを保持します。

</details>
