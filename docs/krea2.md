# Krea 2

## Overview / 概要

This document describes the usage of the Krea 2 (K2) architecture within the Musubi Tuner framework. Krea 2 is a text-to-image generation model based on a single-stream MMDiT, using **Qwen3-VL-4B-Instruct** as the text encoder and the **Qwen-Image VAE** as the autoencoder.

Two DiT checkpoints exist: a **RAW** model (full-step, CFG-based) and a distilled **Turbo** model (few-step, CFG-free). The recommended LoRA workflow is to **train on the RAW model and run inference on the Turbo model** (see [Sample image generation during training](#sample-image-generation-during-training--学習中のサンプル画像生成) and [Inference](#inference--推論)).

This feature is experimental.

> **References:** Official inference code and repository: [krea-ai/krea-2](https://github.com/krea-ai/krea-2). Technical report: [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report).

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのKrea 2 (K2) アーキテクチャの使用法について説明しています。Krea 2はsingle-stream MMDiTをベースとしたテキストから画像を生成するモデルで、テキストエンコーダーに **Qwen3-VL-4B-Instruct**、Autoencoderに **Qwen-Image VAE** を使用します。

DiTのチェックポイントは2種類あります。**RAW** モデル（フルステップ、CFGあり）と、蒸留された **Turbo** モデル（少ステップ、CFGなし）です。推奨されるLoRAのワークフローは、**RAWモデルで学習し、Turboモデルで推論する**ことです（[学習中のサンプル画像生成](#sample-image-generation-during-training--学習中のサンプル画像生成) および [推論](#inference--推論) を参照）。

この機能は実験的なものです。

> **参考:** 公式の推論コードおよびリポジトリ: [krea-ai/krea-2](https://github.com/krea-ai/krea-2)。テクニカルレポート: [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

</details>

## Download the model / モデルのダウンロード

You need to prepare the following models:

- **DiT (RAW)**: The MMDiT transformer model, full-step version. Used for training.
- **DiT (Turbo)** *(optional)*: The distilled few-step version. Used for inference, and optionally for sample image generation during training.
- **VAE**: The **Qwen-Image VAE** (`*.safetensors`). This is the same VAE used by the Qwen-Image integration; if you already have it, you can reuse it.
- **Text Encoder**: **Qwen3-VL-4B-Instruct** as a single `*.safetensors` file (official or ComfyUI key layout). A safetensors **file** is expected here, not a HuggingFace directory. The Comfy-Org single-file weight (`text_encoders/qwen3vl_4b_bf16.safetensors`) is convenient and can be shared with ComfyUI.

The Krea 2 DiT weights are provided as single `*.safetensors` files in the official HuggingFace repositories ([RAW](https://huggingface.co/krea/Krea-2-Raw), [Turbo](https://huggingface.co/krea/Krea-2-Turbo)); each repo also contains a diffusers-compatible checkpoint.

| type | model | file |
|------|-------|------|
| DiT (RAW) | Krea 2 RAW | `raw.safetensors` from https://huggingface.co/krea/Krea-2-Raw |
| DiT (Turbo) | Krea 2 Turbo | `turbo.safetensors` from https://huggingface.co/krea/Krea-2-Turbo |
| VAE | Qwen-Image VAE | `split_files/vae/qwen_image_vae.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI |
| Text Encoder | Qwen3-VL-4B-Instruct | `text_encoders/qwen3vl_4b_bf16.safetensors` from https://huggingface.co/Comfy-Org/Qwen3-VL |

<details>
<summary>日本語</summary>

以下のモデルを準備してください：

- **DiT (RAW)**: MMDiT transformerモデルのフルステップ版。学習に使用します。
- **DiT (Turbo)** *（オプション）*: 蒸留された少ステップ版。推論、および任意で学習中のサンプル画像生成に使用します。
- **VAE**: **Qwen-Image VAE**（`*.safetensors`）。Qwen-Image統合で使用するVAEと同じものです。すでにお持ちであれば再利用できます。
- **Text Encoder**: **Qwen3-VL-4B-Instruct** の単一の `*.safetensors` ファイル（公式またはComfyUIのキー配置）。ここではHuggingFaceのディレクトリではなく、safetensors **ファイル** を指定します。Comfy-Orgの単一ファイル版（`text_encoders/qwen3vl_4b_bf16.safetensors`）が扱いやすく、ComfyUIと共用できます。

Krea 2のDiTの重みは、公式のHuggingFaceリポジトリ（[RAW](https://huggingface.co/krea/Krea-2-Raw)、[Turbo](https://huggingface.co/krea/Krea-2-Turbo)）に単一の `*.safetensors` ファイルとして提供されています（各リポジトリにはdiffusers互換のチェックポイントも含まれます）。ファイル一覧は英語版の表を参照してください: RAWは [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) の `raw.safetensors`、Turboは [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) の `turbo.safetensors` です。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Krea 2. The VAE and latent normalization are identical to Qwen-Image.

```bash
python src/musubi_tuner/krea2_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/qwen_image_vae
```

- Uses `krea2_cache_latents.py`.
- The `--vae` argument is required (Qwen-Image VAE).
- The dataset should be an image dataset. Krea 2 is plain text-to-image only (no control/edit images), so only target image latents are cached.

<details>
<summary>日本語</summary>

latentの事前キャッシングはKrea 2専用のスクリプトを使用します。VAEとlatentの正規化はQwen-Imageと同一です。

- `krea2_cache_latents.py`を使用します。
- `--vae`引数（Qwen-Image VAE）が必要です。
- データセットは画像データセットである必要があります。Krea 2はテキストから画像生成のみ（コントロール/編集画像なし）のため、ターゲット画像のlatentのみがキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script. Krea 2 caches the multi-layer hidden-state stack from Qwen3-VL; only valid (non-padding) tokens are stored (varlen).

```bash
python src/musubi_tuner/krea2_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/qwen3_vl_4b \
    --batch_size 1
```

- Uses `krea2_cache_text_encoder_outputs.py`.
- Requires the `--text_encoder` (Qwen3-VL-4B-Instruct) argument.
- Larger batch sizes require more VRAM. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。Krea 2はQwen3-VLの複数レイヤーのhidden state stackをキャッシュし、有効な（パディングでない）トークンのみを保存します（varlen）。

- `krea2_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder`（Qwen3-VL-4B-Instruct）引数が必要です。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `krea2_train_network.py`. Train on the **RAW** DiT.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \
    --dit path/to/raw_dit_model \
    --vae path/to/qwen_image_vae \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.5 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `krea2_train_network.py`.
- **Requires** specifying `--dit` (RAW model) and `--vae` (Qwen-Image VAE).
- **Requires** specifying `--network_module networks.lora_krea2`.
- `--text_encoder` is only needed if you generate sample images during training (it is not needed for the training step itself, because text encoder outputs are pre-cached).
- Krea 2 uses flow matching. `--timestep_sampling shift` with `--discrete_flow_shift` is a reasonable starting point. The value `2.5` matches the K2 inference time-shift at 1024×1024 (the schedule is resolution-aware: it ranges from about `1.6` at 256×256 to `3.2` at 1280×1280, reaching ~`2.5` at 1024×1024). For varying-resolution training, `--timestep_sampling krea2_shift` reproduces the same resolution-aware schedule per sample, so each timestep is shifted exactly as K2 shifts it at inference (default resolution range 256–1280); no fixed `--discrete_flow_shift` is needed in that case. (`--timestep_sampling flux_shift` is similar but its high end saturates at 1024px instead of 1280px, giving a slightly stronger shift above 256px.) The optimal settings are not yet established; feedback is welcome.
- `--network_dim` / `--network_alpha` of 32 reproduces the model authors' recommended default. See [LoRA target layers](#lora-target-layers--loraの対象レイヤー) below.

<details>
<summary>日本語</summary>

学習は専用のスクリプト`krea2_train_network.py`を使用します。**RAW** のDiTで学習します。コマンド例は英語版を参照してください。

- `krea2_train_network.py`を使用します。
- `--dit`（RAWモデル）と`--vae`（Qwen-Image VAE）を指定する必要があります。
- `--network_module networks.lora_krea2`を指定する必要があります。
- `--text_encoder`は学習中にサンプル画像を生成する場合にのみ必要です（テキストエンコーダー出力は事前キャッシュされるため、学習ステップ自体には不要です）。
- Krea 2はflow matchingを使用します。`--timestep_sampling shift`と`--discrete_flow_shift`の組み合わせが出発点として妥当です。値 `2.5` は1024×1024でのK2推論時のtime-shiftに一致します（このスケジュールは解像度依存で、256×256で約 `1.6`、1280×1280で約 `3.2`、1024×1024で約 `2.5` です）。解像度を変えて学習する場合は、`--timestep_sampling krea2_shift` を使うと同じ解像度依存スケジュールをサンプルごとに再現し、各タイムステップがK2推論時とまったく同じようにシフトされます（デフォルトの解像度レンジ256〜1280）。この場合は固定の `--discrete_flow_shift` は不要です。（`--timestep_sampling flux_shift` も類似ですが、高解像度側が1024px（K2は1280px）で飽和するため、256pxより上ではやや強いshiftになります。）最適な設定はまだ確立されていません。フィードバックをお待ちしています。
- `--network_dim` / `--network_alpha` を32にすると、モデル作者が推奨するデフォルト設定を再現します。下記の[LoRAの対象レイヤー](#lora-target-layers--loraの対象レイヤー)を参照してください。

</details>

### LoRA target layers / LoRAの対象レイヤー

By default, the Krea 2 LoRA targets **all Linear layers** in the DiT (264 layers: attention, MLP, the text-fusion transformer, and the projection MLPs). This matches the model authors' recommended default configuration (rank/alpha 32). The modulation and RMSNorm parameters are raw tensors (not Linear modules), so they are never wrapped — no exclusion is needed.

Because the default already targets everything, both `exclude_patterns` and `include_patterns` are free for you to narrow the target set, passed via `--network_args`:

- **Attention-only** (the authors' "long training run" config — increase rank and focus on the attention projections to preserve prompt adherence):

  ```bash
  --network_args "exclude_patterns=['.*\.mlp\..*','first','last\.linear','tmlp\..*','txtmlp\..*','tproj\.1','txtfusion\..*']"
  ```

  This keeps only the per-block attention projections (`wq`/`wk`/`wv`/`wo`/`gate`, 140 Linears).

- **Arbitrary subset**: drop everything with `exclude_patterns=['.*']`, then add back the layers you want with `include_patterns=[...]`.

<details>
<summary>日本語</summary>

デフォルトでは、Krea 2のLoRAはDiTの **すべてのLinear層**（264層：attention、MLP、text-fusion transformer、projection MLP）を対象とします。これはモデル作者が推奨するデフォルト設定（rank/alpha 32）に一致します。modulationとRMSNormのパラメータは生のテンソル（Linearモジュールではない）なので、対象に含まれません。除外指定は不要です。

デフォルトですべてを対象とするため、`exclude_patterns`と`include_patterns`の両方を対象の絞り込みに自由に使えます。`--network_args`で指定します。

- **Attentionのみ**（作者の「長時間学習」設定。rankを上げてattention projectionに集中し、プロンプト追従性を保つ）: コマンドは英語版を参照。per-blockのattention projection（`wq`/`wk`/`wv`/`wo`/`gate`、140 Linear）のみを残します。
- **任意のサブセット**: `exclude_patterns=['.*']`ですべてを除外し、`include_patterns=[...]`で必要な層を戻します。

</details>

### Memory Optimization / メモリ最適化

- `--fp8_base` and `--fp8_scaled` reduce DiT memory usage. **Both must be specified together** (plain fp8 without scaled is rejected, because it would cast the norms to fp8 and break the model). fp8 is applied to the 28 main blocks only; the text-fusion transformer stays bf16.
- `--blocks_to_swap N` offloads some of the main blocks to CPU. The maximum is **26** (28 blocks − 2).
- `--gradient_checkpointing` is available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

- `--fp8_base`と`--fp8_scaled`でDiTのメモリ使用量を削減します。**両方を同時に指定する必要があります**（scaledなしのplain fp8は、normをfp8にキャストしてモデルを壊すため拒否されます）。fp8は28個のメインブロックのみに適用され、text-fusion transformerはbf16のまま保持されます。
- `--blocks_to_swap N`で一部のメインブロックをCPUにオフロードします。最大値は **26**（28ブロック − 2）です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

</details>

### Attention / Attention

- `--sdpa` for PyTorch's scaled dot product attention (default, no extra dependencies).
- `--flash_attn` for FlashAttention.
- `--sage_attn` for SageAttention.
- `--xformers` for xformers.
- `--split_attn` processes attention in chunks, reducing VRAM usage slightly. Recommended when using any backend other than `--sdpa`.

Krea 2 uses Grouped-Query Attention (48 query heads / 12 key-value heads). SDPA, FlashAttention, and SageAttention support this natively; for xformers the key/value heads are expanded to match (numerically identical).

<details>
<summary>日本語</summary>

- `--sdpa`でPyTorchのscaled dot product attentionを使用（デフォルト、追加の依存ライブラリ不要）。
- `--flash_attn`でFlashAttentionを使用。
- `--sage_attn`でSageAttentionを使用。
- `--xformers`でxformersを使用。
- `--split_attn`を指定すると、attentionを分割して処理し、VRAM使用量をわずかに減らします。`--sdpa`以外のバックエンドを使う場合は指定を推奨します。

Krea 2はGrouped-Query Attention（48 query head / 12 key-value head）を使用します。SDPA、FlashAttention、SageAttentionはこれをネイティブにサポートしており、xformersではkey/value headを拡張して一致させます（数値的に同一）。

</details>

### torch.compile / torch.compile

`--compile` compiles the 28 main SingleStreamBlocks (the heavy, repeated compute and the fp8/LoRA target) for faster training. See [torch.compile documentation](./torch_compile.md). It composes with fp8 and block swap.

<details>
<summary>日本語</summary>

`--compile`で28個のメインSingleStreamBlock（重く繰り返される計算であり、fp8/LoRAの対象）をコンパイルし、学習を高速化します。[torch.compileのドキュメント](./torch_compile.md)を参照してください。fp8やblock swapと併用できます。

</details>

### Sample image generation during training / 学習中のサンプル画像生成

To generate sample images during training, specify `--text_encoder` (Qwen3-VL-4B-Instruct) and the usual `--sample_prompts` / `--sample_every_n_epochs` options. See the [sampling during training documentation](./sampling_during_training.md) for the prompt file format.

By default, samples are generated on the RAW model being trained, using CFG (specify a negative prompt and a CFG scale via `--l` in the sample prompt; CFG-off output is blurry, which is expected for the K2 RAW model).

**Recommended: generate samples on the Turbo model.** Since the recommended workflow is RAW-train → Turbo-infer, you can preview results closer to actual use by sampling on the distilled Turbo model. Pass `--turbo_dit path/to/turbo_dit`. The trained LoRA is applied on top of the Turbo weights automatically (no second network, no merge). When `--turbo_dit` is set, the Turbo schedule is used (fixed `mu = 1.15`); in your sample prompt set CFG off and a low step count, e.g. `--l 1 --s 8`.

```text
A fox in the snow.  --w 1024 --h 1024 --s 8 --l 1 --d 0
```

`--turbo_dit` has two memory modes:

- **Default (streaming)**: the Turbo weights are loaded from disk for each sampling step (re-quantized if fp8) — roughly **no extra steady CPU RAM**, at the cost of per-sample load time.
- **`--turbo_dit_cache` (resident)**: the Turbo weights are quantized once at startup and kept resident in CPU RAM, swapped in for each sample — **faster**, but uses roughly **1× the DiT size in extra CPU RAM** for the whole run.

> **`--turbo_dit` cannot be combined with `--blocks_to_swap`.** Turbo sampling swaps the base weights in place, which is only safe without the block-swap offloader. If you use block swap, omit `--turbo_dit` and sample on the RAW model instead.

<details>
<summary>日本語</summary>

学習中にサンプル画像を生成するには、`--text_encoder`（Qwen3-VL-4B-Instruct）と通常の`--sample_prompts` / `--sample_every_n_epochs`オプションを指定します。プロンプトファイルの形式は[学習中のサンプル生成ドキュメント](./sampling_during_training.md)を参照してください。

デフォルトでは、学習中のRAWモデルでCFGを使用してサンプルが生成されます（サンプルプロンプトでネガティブプロンプトと`--l`によるCFGスケールを指定してください。CFGなしの出力はぼやけますが、これはK2 RAWモデルでは想定通りです）。

**推奨: Turboモデルでサンプルを生成する。** 推奨ワークフローがRAW学習→Turbo推論であるため、蒸留されたTurboモデルでサンプリングすると、実利用に近い結果をプレビューできます。`--turbo_dit path/to/turbo_dit`を指定します。学習中のLoRAは自動的にTurboの重みの上に適用されます（2つ目のネットワークもマージも不要）。`--turbo_dit`指定時はTurboのスケジュール（固定`mu = 1.15`）が使われます。サンプルプロンプトではCFGをオフにし、少ないステップ数を指定してください（例: `--l 1 --s 8`）。

`--turbo_dit`には2つのメモリモードがあります。

- **デフォルト（ストリーミング）**: Turboの重みをサンプリングステップごとにディスクから読み込みます（fp8の場合は再量子化）。**定常のCPU RAM増加はほぼゼロ**ですが、サンプルごとの読み込み時間がかかります。
- **`--turbo_dit_cache`（常駐）**: Turboの重みを起動時に一度量子化してCPU RAMに常駐させ、サンプルごとにスワップインします。**高速**ですが、実行中ずっと **DiTサイズの約1倍** のCPU RAMを追加で使用します。

> **`--turbo_dit`は`--blocks_to_swap`と併用できません。** Turboサンプリングはベースの重みをその場で入れ替えるため、block swapのオフローダーがない場合にのみ安全です。block swapを使う場合は`--turbo_dit`を省略し、RAWモデルでサンプリングしてください。

</details>

## Inference / 推論

Inference uses a dedicated script `krea2_generate_image.py`. For the recommended workflow, run inference on the **Turbo** model.

**RAW model inference:**

```bash
python src/musubi_tuner/krea2_generate_image.py \
    "A fox in the snow." \
    --dit path/to/raw_dit_model \
    --vae path/to/qwen_image_vae \
    --text_encoder path/to/qwen3_vl_4b \
    --steps 28 --guidance_scale 5.5 \
    --width 1024 --height 1024 \
    --attn_mode torch \
    --seed 0 --save_path path/to/save/dir \
    --lora_weight path/to/lora.safetensors --lora_multiplier 1.0
```

**Turbo model inference (recommended):**

```bash
python src/musubi_tuner/krea2_generate_image.py \
    "A fox in the snow." \
    --dit path/to/turbo_dit_model \
    --vae path/to/qwen_image_vae \
    --text_encoder path/to/qwen3_vl_4b \
    --steps 8 --guidance_scale 1 --mu 1.15 \
    --width 1024 --height 1024 \
    --attn_mode torch \
    --seed 0 --save_path path/to/save/dir \
    --lora_weight path/to/lora.safetensors --lora_multiplier 1.0
```

- Uses `krea2_generate_image.py`. There are three input modes: a single positional prompt (above), `--from_file <path>` (one prompt per line), and `--interactive` (read prompts from the console). Exactly one must be given.
  - In `--from_file` / `--interactive`, each line may carry per-prompt overrides as `--<opt> <value>` after the prompt text: `--w` (width), `--h` (height), `--s` (steps), `--d` (seed), `--g` / `--l` (guidance_scale), `--n` (negative prompt), `--y1`, `--y2`, `--mu`, `--i` (num images). Example: `A fox in the snow. --w 1280 --h 768 --s 8 --l 1 --d 0`. Blank lines and lines starting with `#` are skipped. Anything not overridden falls back to the command-line value. `--bell` rings the terminal bell after each prompt (interactive) or at the end.
- **Requires** `--dit`, `--vae` (Qwen-Image VAE), and `--text_encoder` (Qwen3-VL-4B-Instruct).
- `--steps` defaults to 28 (use ~8 for Turbo).
- `--guidance_scale` is the classifier-free guidance scale, default 5.5. Values `<= 1` disable CFG (use `--guidance_scale 1` for the Turbo model; no negative prompt needed). `--negative_prompt` is used only when `--guidance_scale > 1`. Krea 2 uses the standard CFG form `uncond + scale * (cond - uncond)`, like the other architectures here. **Note:** the official Krea 2 reference uses a "guidance" value with a different baseline (`uncond` at `0`), related by `guidance_scale = guidance + 1` — so the official default `guidance 4.5` corresponds to `--guidance_scale 5.5`.
- Timestep-shift `mu`: by default it is resolution-aware (interpolated between `--y1` at min resolution and `--y2` at max). For the Turbo model, pin a constant with `--mu 1.15`.
- `--attn_mode` selects the attention backend: `torch` (SDPA, default), `flash`, `sageattn`, `xformers`. Add `--split_attn` for the non-sdpa backends (required for `xformers` with GQA).
- `--width` / `--height` default to 1024. `--num-images` generates multiple images (image *i* uses `--seed` + *i*).
- `--save_path` is a **directory** (required, created if missing); file names are auto-generated as `<timestamp>_<seed>.png`.
- `--seed` is the base seed; image *i* uses `--seed` + *i*. **If omitted, a random seed is used** (logged, and reflected in the file name). In `--from_file` / `--interactive`, prompts without a per-line `--d` each draw a fresh random seed.
- LoRA loading: `--lora_weight` (one or more) and `--lora_multiplier`. LoRA is merged into the base DiT weights at load time (the only correct route under fp8).
- **Memory-efficient inference (fits a 24GB card):**
  - `--fp8_scaled` quantizes the 28 main blocks to dynamic scaled fp8 at load time (K2 supports only scaled fp8; the text-fusion transformer stays bf16). Roughly halves DiT weight memory.
  - `--blocks_to_swap N` offloads `N` of the main blocks to CPU (max **26** = 28 − 2), trading speed for VRAM. Composes with `--fp8_scaled`.
  - `--use_pinned_memory_for_block_swap` uses pinned host memory for faster H2D copies (more host RAM).
  - `--block_swap_h2d_only` streams blocks Host→Device only (keeping a CPU master, no device→host copy), which is always safe at inference since the base weights are frozen. `--block_swap_ring_size N` sets the number of GPU ring buffers (2 = transfer/compute overlap, 1 = minimal memory).
  - **Memory model:** the DiT stays resident on the GPU (with block swap as needed) for the whole run, while the text encoder (Qwen3-VL-4B, ~8GB) and the VAE shuttle between CPU and GPU. The encoder is kept on CPU and moved to the GPU only to encode each prompt; the VAE is kept on CPU and moved to the GPU only to decode, then moved back. So the headroom for encoding/decoding comes from running the DiT under `--fp8_scaled` and/or `--blocks_to_swap` — not from evacuating the ~24GB DiT to host RAM. (Block swap and/or fp8 is therefore effectively required to fit a 24GB card; without it the DiT alone leaves no room for the decode.)
  - `--text_encoder_cpu` encodes prompts on CPU instead of moving the encoder to the GPU. Use it when the encoder (on the GPU, alongside the resident DiT) does not fit; it is slower but keeps the encode off the GPU.

<details>
<summary>日本語</summary>

推論は専用のスクリプト`krea2_generate_image.py`を使用します。推奨ワークフローでは **Turbo** モデルで推論します。コマンド例は英語版を参照してください。

- `krea2_generate_image.py`を使用します。入力モードは3種類あります: 単一の位置引数プロンプト（上記）、`--from_file <path>`（1行1プロンプト）、`--interactive`（コンソールから入力）。いずれか1つを指定します。
  - `--from_file` / `--interactive`では、各行のプロンプト本文の後ろに`--<opt> <value>`形式でプロンプトごとの上書きを記述できます: `--w`（幅）、`--h`（高さ）、`--s`（ステップ数）、`--d`（シード）、`--g` / `--l`（guidance_scale）、`--n`（ネガティブプロンプト）、`--y1`、`--y2`、`--mu`、`--i`（生成枚数）。例: `A fox in the snow. --w 1280 --h 768 --s 8 --l 1 --d 0`。空行と`#`で始まる行はスキップされます。上書きされなかった項目はコマンドライン引数の値が使われます。`--bell`で各プロンプト後（対話モード）または最後（その他のモード）に端末のベルを鳴らします。
- `--dit`、`--vae`（Qwen-Image VAE）、`--text_encoder`（Qwen3-VL-4B-Instruct）が必要です。
- `--steps`のデフォルトは28です（Turboでは約8）。
- `--guidance_scale`はclassifier-freeガイダンスのスケールで、デフォルトは5.5です。`<= 1` でCFGを無効化します（Turboモデルでは`--guidance_scale 1`。ネガティブプロンプト不要）。`--negative_prompt`は`--guidance_scale > 1`のときのみ使用されます。Krea 2は他のアーキテクチャと同じく標準のCFG式 `uncond + scale * (cond - uncond)` を使用します。**注意:** 公式のKrea 2 referenceは基準点の異なる「guidance」値（`0` で `uncond`）を使っており、関係式は `guidance_scale = guidance + 1` です。したがって公式デフォルトの `guidance 4.5` は `--guidance_scale 5.5` に対応します。
- Timestep-shiftの`mu`: デフォルトは解像度依存（最小解像度の`--y1`と最大解像度の`--y2`の間で補間）です。Turboモデルでは`--mu 1.15`で固定値を指定してください。
- `--attn_mode`でattentionバックエンドを選択します: `torch`（SDPA、デフォルト）、`flash`、`sageattn`、`xformers`。sdpa以外のバックエンドでは`--split_attn`を追加してください（`xformers` + GQAでは必須）。
- `--width` / `--height`のデフォルトは1024です。`--num-images`で複数画像を生成します（画像 *i* は`--seed` + *i* を使用）。
- `--save_path`は保存先の**ディレクトリ**です（必須、なければ作成）。ファイル名は`<タイムスタンプ>_<seed>.png`の形式で自動生成されます。
- `--seed`はベースシードで、画像 *i* は`--seed` + *i* を使用します。**省略した場合はランダムなシードが使われます**（ログに出力され、ファイル名にも反映されます）。`--from_file` / `--interactive`では、行ごとの`--d`がないプロンプトはそれぞれ新しいランダムシードを引きます。
- LoRAの読み込み: `--lora_weight`（1つ以上）と`--lora_multiplier`。LoRAは読み込み時にベースのDiT重みにマージされます（fp8でも正しく動作する唯一の方法）。
- **省メモリ推論（24GBに収まります）:**
  - `--fp8_scaled`で28個のメインブロックを読み込み時に動的スケールfp8に量子化します（K2はscaled fp8のみ対応。text-fusion transformerはbf16のまま）。DiTの重みメモリがおよそ半減します。
  - `--blocks_to_swap N`で`N`個のメインブロックをCPUにオフロードします（最大 **26** = 28 − 2）。速度と引き換えにVRAMを削減します。`--fp8_scaled`と併用できます。
  - `--use_pinned_memory_for_block_swap`でpinnedホストメモリを使い、H2Dコピーを高速化します（ホストRAMを多く使用）。
  - `--block_swap_h2d_only`はブロックをHost→Deviceのみでストリーミングします（CPUマスターを保持し、device→hostコピーを行わない）。推論ではベース重みが凍結されているため常に安全です。`--block_swap_ring_size N`でGPUリングバッファ数を設定します（2で転送と計算をオーバーラップ、1で最小メモリ）。
  - **メモリモデル:** DiTは実行中ずっとGPUに常駐させ（必要に応じてblock swap）、テキストエンコーダ（Qwen3-VL-4B、約8GB）とVAEがCPUとGPUを行き来します。エンコーダはCPUに置き、各プロンプトのエンコード時のみGPUへ移動します。VAEもCPUに置き、デコード時のみGPUへ移動してから戻します。したがってエンコード/デコードのためのVRAMの余裕は、`--fp8_scaled`や`--blocks_to_swap`でDiTを動かすことから生まれます（約24GBのDiTをホストRAMへ退避させるのではありません）。このため24GBに収めるにはblock swapおよび/またはfp8が実質必須です（使わない場合、DiTだけでデコードの余地が残りません）。
  - `--text_encoder_cpu`はエンコーダをGPUに移動せずCPUでエンコードします。常駐DiTと並んでエンコーダがGPUに載りきらない場合に使用します。低速ですがエンコードをGPUの外に出せます。

</details>
