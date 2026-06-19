# Ideogram 4

## Overview / 概要

This document describes how to train and run inference for Ideogram 4 within the Musubi Tuner framework. Ideogram 4 is a text-to-image model. Its main components are:

- **DiT**: a 34-layer single-stream transformer.
- **Text Encoder**: Qwen3-VL 8B. Hidden states from multiple layers are concatenated into a 53,248-channel conditioning sequence.
- **VAE**: the Flux2 KL-VAE.
- **Asymmetric CFG**: inference uses a separate conditional DiT and unconditional DiT.

This adapter targets the Comfy-Org single-file component layout. The model license is non-commercial; read and accept the relevant terms before downloading or using the weights.

Ideogram 4 is distributed only in quantized form (FP8 and NVFP4); there are no official BF16/FP16 DiT weights. The DiT is therefore always loaded from a pre-quantized FP8 checkpoint — for both inference and LoRA training — and kept in FP8 as the frozen base. The FP8 weights are dequantized on the fly to the compute dtype, and any LoRA modules run in the compute dtype, so this is the normal (and only) operating mode rather than an optional memory optimization.

Many options are shared with other architectures and can be found via `--help`. Refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed. This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのIdeogram 4の学習・推論方法について説明します。Ideogram 4はテキストから画像を生成するモデルです。主な構成要素は次の通りです。

- **DiT**: 34層のsingle-stream transformer。
- **Text Encoder**: Qwen3-VL 8B。複数レイヤーの隠れ状態を連結し、53,248チャンネルの条件付けシーケンスを作ります。
- **VAE**: Flux2 KL-VAE。
- **非対称CFG**: 推論ではconditional DiTとunconditional DiTを別々に使用します。

このアダプターはComfy-Orgの単一ファイル形式を対象としています。モデルのライセンスは非商用です。重みをダウンロード・使用する前に、関連する規約を読んで同意してください。

Ideogram 4は量子化された形式（FP8およびNVFP4）でのみ配布されており、公式のBF16/FP16 DiT重みは存在しません。そのためDiTは、推論・LoRA学習のどちらでも常に事前量子化されたFP8チェックポイントから読み込まれ、凍結したベースとしてFP8のまま保持されます。FP8重みはオンザフライでcompute dtypeに逆量子化され、LoRAモジュールもcompute dtypeで動作します。つまりこれはオプションのメモリ最適化ではなく、通常の（そして唯一の）動作モードです。

多くのオプションは他のアーキテクチャと共通で、`--help`で確認できます。必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

Download the component files yourself from https://huggingface.co/Comfy-Org/Ideogram-4 and pass local paths to the scripts. The scripts do not accept `--repo_id` and do not download Comfy-Org weights for you.

Required files:

- `diffusion_models/ideogram4_fp8_scaled.safetensors` — conditional DiT
- `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors` — unconditional DiT (for inference / asymmetric CFG)
- `text_encoders/qwen3vl_8b_fp8_scaled.safetensors` — Qwen3-VL text encoder
- `vae/flux2-vae.safetensors` — Flux2 VAE

The tokenizer is downloaded automatically from `Qwen/Qwen3-VL-8B-Instruct`; the text-encoder config is bundled in this repository. There are no tokenizer/config CLI arguments.

<details>
<summary>日本語</summary>

コンポーネントファイルは https://huggingface.co/Comfy-Org/Ideogram-4 から自分でダウンロードし、ローカルパスをスクリプトに渡してください。スクリプトは`--repo_id`を受け付けず、Comfy-Orgの重みを自動でダウンロードすることはありません。

必要なファイル:

- `diffusion_models/ideogram4_fp8_scaled.safetensors` — conditional DiT
- `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors` — unconditional DiT（推論／非対称CFG用）
- `text_encoders/qwen3vl_8b_fp8_scaled.safetensors` — Qwen3-VL text encoder
- `vae/flux2-vae.safetensors` — Flux2 VAE

トークナイザーは`Qwen/Qwen3-VL-8B-Instruct`から自動でダウンロードされます。text encoderのconfigはこのリポジトリ内に同梱されています。トークナイザー/config用のCLI引数はありません。

</details>

## Caption format / キャプション形式

Ideogram 4 was trained with structured JSON captions (top-level keys such as `high_level_description`, `style_description`, `compositional_deconstruction`). A built-in verifier can check this structure.

- **Latent / text caching**: plain text captions are accepted by default. Add `--validate_caption_structure` to verify official structured JSON captions, and combine it with `--warn_on_caption_issues` to warn instead of failing.
- **Inference**: the prompt is always verified against the structured JSON format. Plain text or malformed JSON raises an error unless you pass `--warn_on_caption_issues`, which downgrades the failure to a warning.

<details>
<summary>日本語</summary>

Ideogram 4は構造化されたJSONキャプション（`high_level_description`、`style_description`、`compositional_deconstruction`などのトップレベルキー）で学習されています。組み込みのverifierでこの構造を検証できます。

- **latent／textキャッシュ時**: プレーンテキストのキャプションは既定で受け付けられます。`--validate_caption_structure`を付けると公式の構造化JSONキャプションを検証し、`--warn_on_caption_issues`と併用すると失敗の代わりに警告にできます。
- **推論時**: プロンプトは常に構造化JSON形式として検証されます。プレーンテキストや不正なJSONはエラーになりますが、`--warn_on_caption_issues`を付けると失敗が警告に降格されます。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

```powershell
python src/musubi_tuner/ideogram4_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae path\to\flux2-vae.safetensors `
  --vae_dtype bfloat16
```

- Uses `ideogram4_cache_latents.py`. The dataset should be an image dataset.
- `--vae_dtype` sets the VAE compute dtype (default `bfloat16`).
- The latent cache stores the raw patchified VAE encoder mean in `(patch_i, patch_j, latent_channel)` order. Training applies the Ideogram 4 latent shift/scale once, matching the official pipeline.

<details>
<summary>日本語</summary>

- `ideogram4_cache_latents.py`を使用します。データセットは画像データセットである必要があります。
- `--vae_dtype`はVAEのcompute dtypeを指定します（既定は`bfloat16`）。
- latentキャッシュは、patchify済みのVAE encoder meanを`(patch_i, patch_j, latent_channel)`の順で保存します。学習時にIdeogram 4のlatent shift/scaleを一度だけ適用し、公式パイプラインと一致させます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

```powershell
python src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py `
  --dataset_config path\to\dataset.toml `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --text_cache_dtype bf16
```

- Uses `ideogram4_cache_text_encoder_outputs.py`. Requires `--text_encoder`.
- See [Caption format](#caption-format--キャプション形式) for `--validate_caption_structure` / `--warn_on_caption_issues`.
- The text cache is large because each token stores 53,248 channels. Approximate BF16 cost:
  - 128 tokens: 13.6 MB/image, 13.6 GB/1k images
  - 256 tokens: 27.3 MB/image, 27.3 GB/1k images
  - 512 tokens: 54.5 MB/image, 54.5 GB/1k images
  - 2048 tokens: 218.1 MB/image, 218.1 GB/1k images
- `--text_cache_dtype fp8_e4m3fn` halves the disk cost but is an explicit quality/precision tradeoff.

<details>
<summary>日本語</summary>

- `ideogram4_cache_text_encoder_outputs.py`を使用します。`--text_encoder`が必要です。
- `--validate_caption_structure`／`--warn_on_caption_issues`については[キャプション形式](#caption-format--キャプション形式)を参照してください。
- 各トークンが53,248チャンネルを保存するため、テキストキャッシュは大きくなります。BF16でのおおよそのサイズは英語版を参照してください。
- `--text_cache_dtype fp8_e4m3fn`を指定するとディスク使用量は半分になりますが、明確な品質/精度のトレードオフがあります。

</details>

## Training / 学習

Training uses a dedicated script `ideogram4_train_network.py`.

```powershell
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/ideogram4_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --network_module networks.lora_ideogram4 `
  --output_dir outputs\i4_lora `
  --output_name my_i4_lora `
  --mixed_precision bf16 `
  --sdpa
```

- Uses `ideogram4_train_network.py` and **requires** `--network_module networks.lora_ideogram4`.
- LoRA v1 trains only the conditional transformer. It targets `attention.qkv`, `attention.o`, and `feed_forward.w1/w2/w3` inside `Ideogram4TransformerBlock`.
- Merging LoRA back into the FP8 base weights is not supported.
- `--dit_dtype` sets the DiT compute dtype (default `bfloat16`).
- Ideogram 4 uses a plain MSE flow-matching loss; `--weighting_scheme` other than `none` is rejected.
- `--log_loss_stats` logs prediction/target diagnostics during training.
- Attention backends use the shared flags: `--sdpa` (default), `--flash_attn`, `--xformers`, with `--split_attn`. SageAttention is not supported because the head dimension (256) is outside its int8/fp8 kernels.

<details>
<summary>日本語</summary>

学習は専用のスクリプト`ideogram4_train_network.py`を使用します。コマンド例は英語版を参照してください。

- `ideogram4_train_network.py`を使用し、`--network_module networks.lora_ideogram4`の指定が**必要**です。
- LoRA v1はconditional transformerのみを学習します。対象は`Ideogram4TransformerBlock`内の`attention.qkv`、`attention.o`、`feed_forward.w1/w2/w3`です。
- LoRAをFP8ベース重みにマージし直すことはサポートされていません。
- `--dit_dtype`はDiTのcompute dtypeを指定します（既定は`bfloat16`）。
- Ideogram 4は単純なMSEのflow-matching lossを使用します。`none`以外の`--weighting_scheme`はエラーになります。
- `--log_loss_stats`は学習中にprediction/targetの診断情報をログ出力します。
- attentionのバックエンドは共通フラグ（`--sdpa`（既定）、`--flash_attn`、`--xformers`、`--split_attn`）を使用します。SageAttentionはhead次元（256）がint8/fp8カーネルの対応外のため使用できません。

</details>

### Timestep sampling / タイムステップサンプリング

Ideogram 4 training defaults to `--timestep_sampling ideogram4_shift`, a resolution-aware logit-normal timestep sampler aligned with the official pipeline. You normally do not need to change it.

<details>
<summary>日本語</summary>

Ideogram 4の学習は既定で`--timestep_sampling ideogram4_shift`を使用します。これは公式パイプラインに合わせた、解像度依存のlogit-normalタイムステップサンプラーです。通常は変更する必要はありません。

</details>

### Memory Optimization / メモリ最適化

- `--blocks_to_swap` offloads some DiT blocks to CPU. The maximum is 33.
- `--gradient_checkpointing` (and `--gradient_checkpointing_cpu_offload`) reduce VRAM usage. See the [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- The DiT base is already FP8 (see [Overview](#overview--概要)), so no separate `--fp8_base` flag is needed for it.

<details>
<summary>日本語</summary>

- `--blocks_to_swap`は一部のDiTブロックをCPUにオフロードします。最大値は33です。
- `--gradient_checkpointing`（および`--gradient_checkpointing_cpu_offload`）でVRAM使用量を削減できます。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。
- DiTベースは既にFP8です（[概要](#overview--概要)参照）。そのため別途`--fp8_base`フラグは不要です。

</details>

### Sampling during training / 学習中のサンプル生成

Sampling during training requires the remaining local components:

```powershell
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/ideogram4_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --vae path\to\flux2-vae.safetensors `
  --network_module networks.lora_ideogram4 `
  --sample_prompts path\to\prompts.txt `
  --sample_every_n_steps 500 `
  --sampler_preset V4_DEFAULT_20 `
  --initial_sigma 1.004 `
  --mixed_precision bf16 `
  --sdpa
```

- `--text_encoder` and `--vae` are only needed when sampling.
- `--sampler_preset` selects the step schedule (see [Inference](#inference--推論) for the preset list); `--initial_sigma` overrides the first denoising sigma (default `1.004`).
- Sampling during LoRA training defaults to single-DiT CFG, because the LoRA adapter is attached only to the conditional DiT. If you pass `--unconditional_dit`, it is ignored for training samples unless you also pass `--use_unconditional_dit_for_lora_sampling` to opt into the official asymmetric CFG path.

<details>
<summary>日本語</summary>

学習中のサンプル生成には、残りのローカルコンポーネントが必要です。コマンド例は英語版を参照してください。

- `--text_encoder`と`--vae`はサンプル生成時のみ必要です。
- `--sampler_preset`はステップスケジュールを選択します（プリセット一覧は[推論](#inference--推論)を参照）。`--initial_sigma`は最初のデノイジングsigmaを上書きします（既定は`1.004`）。
- LoRAアダプターはconditional DiTのみに付与されるため、LoRA学習中のサンプル生成は既定でsingle-DiT CFGになります。`--unconditional_dit`を渡しても、`--use_unconditional_dit_for_lora_sampling`を併用して公式の非対称CFGパスを明示的に有効化しない限り、学習サンプルでは無視されます。

</details>

## Inference / 推論

Inference uses a dedicated script `ideogram4_generate_image.py`.

```powershell
python src/musubi_tuner/ideogram4_generate_image.py `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --unconditional_dit path\to\ideogram4_unconditional_fp8_scaled.safetensors `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --vae path\to\flux2-vae.safetensors `
  --prompt "A structured JSON prompt" `
  --image_size 1024 1024 `
  --sampler_preset V4_DEFAULT_20 `
  --initial_sigma 1.004 `
  --save_path outputs\ideogram4.png
```

Sampler presets (`--sampler_preset`):

- `V4_QUALITY_48`: 45 main steps at guidance 7, then 3 polish steps at guidance 3
- `V4_DEFAULT_20`: 18 main steps at guidance 7, then 2 polish steps at guidance 3
- `V4_TURBO_12`: 11 main steps at guidance 7, then 1 polish step at guidance 3

Notes:

- Ideogram 4 v1 uses the official asymmetric CFG path with the conditional and unconditional DiTs. `--negative_prompt` is ignored.
- `--initial_sigma` overrides the first denoising sigma (default `1.004`).
- `--dtype` sets the compute dtype for the models (default `bfloat16`).
- `--prompt` is verified as structured JSON; see [Caption format](#caption-format--キャプション形式).
- `--disable_numpy_memmap` disables numpy memmap while loading safetensors (higher RAM, sometimes faster loading).

### LoRA / LoRA

To apply trained LoRA weights, add `--lora_weight path\to\lora.safetensors` (repeatable) with optional `--lora_multiplier` (one value per weight, default `1.0`), and optional `--include_patterns` / `--exclude_patterns` (one regex per weight). The LoRA is attached to the conditional DiT as a forward hook rather than merged into the weights, because the FP8 base cannot have LoRA merged back into it. This means generation results may differ from tools that merge LoRA into the weights (e.g. ComfyUI).

### Attention / Attention

`--attn_mode` selects the attention backend (`torch`/`sdpa` (default), `flash`, `sageattn`, `xformers`); add `--split_attn` to process each sample's attention separately. `flash`/`sageattn`/`xformers` require the respective package to be installed and can be faster, but the numerics differ slightly from `torch`. (`sageattn` does not support the head dimension of 256 and will fail.)

<details>
<summary>日本語</summary>

推論は専用のスクリプト`ideogram4_generate_image.py`を使用します。コマンド例とサンプラープリセット一覧は英語版を参照してください。

- Ideogram 4 v1はconditional/unconditional DiTを用いた公式の非対称CFGパスを使用します。`--negative_prompt`は無視されます。
- `--initial_sigma`は最初のデノイジングsigmaを上書きします（既定は`1.004`）。
- `--dtype`はモデルのcompute dtypeを指定します（既定は`bfloat16`）。
- `--prompt`は構造化JSONとして検証されます。[キャプション形式](#caption-format--キャプション形式)を参照してください。
- `--disable_numpy_memmap`はsafetensors読み込み時のnumpy memmapを無効化します（RAM使用量は増えますが、読み込みが速くなる場合があります）。

#### LoRA

学習済みLoRA重みを適用するには、`--lora_weight path\to\lora.safetensors`（繰り返し指定可）を追加し、必要に応じて`--lora_multiplier`（重みごとに1値、既定`1.0`）、`--include_patterns`／`--exclude_patterns`（重みごとに1つのregex）を指定します。FP8ベースにはLoRAをマージし直せないため、LoRAは重みへのマージではなくconditional DiTへのforward hookとして付与されます。このため、LoRAを重みにマージするツール（例: ComfyUI）とは生成結果が異なる場合があります。

#### Attention

`--attn_mode`でattentionのバックエンドを選択します（`torch`/`sdpa`（既定）、`flash`、`sageattn`、`xformers`）。`--split_attn`を追加すると各サンプルのattentionを個別に処理します。`flash`/`sageattn`/`xformers`はそれぞれのパッケージのインストールが必要で、高速化が期待できますが、数値は`torch`とわずかに異なります。（`sageattn`はhead次元256に対応しておらず失敗します。）

</details>
