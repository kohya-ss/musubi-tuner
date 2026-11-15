> 📝 Click on the language section to expand / 言語をクリックして展開

# LoHa / LoKr (LyCORIS)

## Overview / 概要

In addition to standard LoRA, Musubi Tuner supports **LoHa** (Low-rank Hadamard Product) and **LoKr** (Low-rank Kronecker Product) as alternative parameter-efficient fine-tuning methods. These are based on techniques from the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) project.

- **LoHa**: Represents weight updates as a Hadamard (element-wise) product of two low-rank matrices. Reference: [FedPara (arXiv:2108.06098)](https://arxiv.org/abs/2108.06098)
- **LoKr**: Represents weight updates as a Kronecker product with optional low-rank decomposition. Reference: [LoKr (arXiv:2309.14859)](https://arxiv.org/abs/2309.14859)

The algorithms and recommended settings are described in the [LyCORIS documentation](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md) and [guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md).

Both methods target Linear layers only (Conv2d layers are not supported in this implementation).

This feature is experimental.

<details>
<summary>日本語</summary>

Musubi Tunerでは、標準的なLoRAに加え、代替のパラメータ効率の良いファインチューニング手法として **LoHa**（Low-rank Hadamard Product）と **LoKr**（Low-rank Kronecker Product）をサポートしています。これらは [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) プロジェクトの手法に基づいています。

- **LoHa**: 重みの更新を2つの低ランク行列のHadamard積（要素ごとの積）で表現します。参考文献: [FedPara (arXiv:2108.06098)](https://arxiv.org/abs/2108.06098)
- **LoKr**: 重みの更新をKronecker積と、オプションの低ランク分解で表現します。参考文献: [LoKr (arXiv:2309.14859)](https://arxiv.org/abs/2309.14859)

アルゴリズムと推奨設定は[LyCORISのアルゴリズム解説](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md)と[ガイドライン](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md)を参照してください。

いずれもLinear層のみを対象としています（Conv2d層はこの実装ではサポートしていません）。

この機能は実験的なものです。

</details>

## Acknowledgments / 謝辞

The LoHa and LoKr implementations in Musubi Tuner are based on the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) project by [KohakuBlueleaf](https://github.com/KohakuBlueleaf). We would like to express our sincere gratitude for the excellent research and open-source contributions that made this implementation possible.

<details>
<summary>日本語</summary>

Musubi TunerのLoHaおよびLoKrの実装は、[KohakuBlueleaf](https://github.com/KohakuBlueleaf)氏による[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)プロジェクトに基づいています。この実装を可能にしてくださった素晴らしい研究とオープンソースへの貢献に心から感謝いたします。

</details>

## Supported architectures / 対応アーキテクチャ

LoHa and LoKr automatically detect the model architecture and apply appropriate default settings. The following architectures are supported:

- HunyuanVideo
- HunyuanVideo 1.5
- Wan 2.1/2.2
- FramePack
- FLUX.1 Kontext / FLUX.2
- Qwen-Image series
- Z-Image
- Krea 2

Kandinsky5 is **not supported** with LoHa/LoKr (it requires special handling that is incompatible with automatic architecture detection).

Each architecture has its own default `exclude_patterns` to skip non-trainable modules (e.g., modulation layers, normalization layers). These are applied automatically when using LoHa/LoKr.

<details>
<summary>日本語</summary>

LoHaとLoKrは、モデルのアーキテクチャを自動で検出し、適切なデフォルト設定を適用します。以下のアーキテクチャに対応しています:

- HunyuanVideo
- HunyuanVideo 1.5
- Wan 2.1/2.2
- FramePack
- FLUX.1 Kontext / FLUX.2
- Qwen-Image系
- Z-Image
- Krea 2

Kandinsky5はLoHa/LoKrに **対応していません**（自動アーキテクチャ検出と互換性のない特殊な処理が必要です）。

各アーキテクチャには、学習対象外のモジュール（modulation層、normalization層など）をスキップするデフォルトの `exclude_patterns` が設定されています。LoHa/LoKr使用時にはこれらが自動的に適用されます。

</details>

## Training / 学習

To use LoHa or LoKr, change the `--network_module` argument in your training command. All other training options (dataset config, optimizer, etc.) remain the same as LoRA.

<details>
<summary>日本語</summary>

LoHaまたはLoKrを使用するには、学習コマンドの `--network_module` 引数を変更します。その他の学習オプション（データセット設定、オプティマイザなど）はLoRAと同じです。

</details>

### LoHa

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py \
    --dit path/to/dit \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --network_module networks.loha --network_dim 32 --network_alpha 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name my-loha
```

### LoKr

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py \
    --dit path/to/dit \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --network_module networks.lokr --network_dim 32 --network_alpha 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name my-lokr
```

Replace `hv_train_network.py` with the appropriate training script for your architecture (e.g., `wan_train_network.py`, `fpack_train_network.py`, etc.).

<details>
<summary>日本語</summary>

`hv_train_network.py` の部分は、お使いのアーキテクチャに対応する学習スクリプト（`wan_train_network.py`, `fpack_train_network.py` など）に置き換えてください。

</details>

### Common training options / 共通の学習オプション

The following `--network_args` options are available for both LoHa and LoKr, same as LoRA:

| Option | Description |
|---|---|
| `verbose=True` | Display detailed information about the network modules |
| `rank_dropout=0.1` | Apply dropout to the rank dimension during training |
| `module_dropout=0.1` | Randomly skip entire modules during training |
| `exclude_patterns=[r'...']` | Exclude modules matching the regex patterns (in addition to architecture defaults) |
| `include_patterns=[r'...']` | Include only modules matching the regex patterns |

See [Advanced configuration](advanced_config.md) for details on how to specify `network_args`.

<details>
<summary>日本語</summary>

以下の `--network_args` オプションは、LoRAと同様にLoHaとLoKrの両方で使用できます:

| オプション | 説明 |
|---|---|
| `verbose=True` | ネットワークモジュールの詳細情報を表示 |
| `rank_dropout=0.1` | 学習時にランク次元にドロップアウトを適用 |
| `module_dropout=0.1` | 学習時にモジュール全体をランダムにスキップ |
| `exclude_patterns=[r'...']` | 正規表現パターンに一致するモジュールを除外（アーキテクチャのデフォルトに追加） |
| `include_patterns=[r'...']` | 正規表現パターンに一致するモジュールのみを対象とする |

`network_args` の指定方法の詳細は [高度な設定](advanced_config.md) を参照してください。

</details>

### LoKr-specific option: `factor` / LoKr固有のオプション: `factor`

LoKr decomposes weight dimensions using factorization. The `factor` option controls how dimensions are split:

- `factor=-1` (default): Automatically find balanced factors. For example, dimension 512 is split into (16, 32).
- `factor=N` (positive integer): Force factorization using the specified value. For example, `factor=4` splits dimension 512 into (4, 128).

```bash
--network_args "factor=4"
```

When `network_dim` (rank) is large enough relative to the factorized dimensions, LoKr uses a full matrix instead of a low-rank decomposition for the second factor. A warning will be logged in this case.

<details>
<summary>日本語</summary>

LoKrは重みの次元を因数分解して分割します。`factor` オプションでその分割方法を制御します:

- `factor=-1`（デフォルト）: バランスの良い因数を自動的に見つけます。例えば、次元512は(16, 32)に分割されます。
- `factor=N`（正の整数）: 指定した値で因数分解します。例えば、`factor=4` は次元512を(4, 128)に分割します。

```bash
--network_args "factor=4"
```

`network_dim`（ランク）が因数分解された次元に対して十分に大きい場合、LoKrは第2因子に低ランク分解ではなくフル行列を使用します。その場合、警告がログに出力されます。

</details>

## How LoHa and LoKr work / LoHaとLoKrの仕組み

### LoHa

LoHa represents the weight update as a Hadamard (element-wise) product of two low-rank matrices:

```
ΔW = (W1a × W1b) ⊙ (W2a × W2b)
```

where `W1a`, `W1b`, `W2a`, `W2b` are low-rank matrices with rank `network_dim`. This means LoHa has roughly **twice the number of trainable parameters** compared to LoRA at the same rank, but can capture more complex weight structures due to the element-wise product.

### LoKr

LoKr represents the weight update using a Kronecker product:

```
ΔW = W1 ⊗ W2    (where W2 = W2a × W2b in low-rank mode)
```

The original weight dimensions are factorized (e.g., a 512×512 weight might be split so that W1 is 16×16 and W2 is 32×32). W1 is always a full matrix (small), while W2 can be either low-rank decomposed or a full matrix depending on the rank setting. LoKr tends to produce **smaller models** compared to LoRA at the same rank.

<details>
<summary>日本語</summary>

### LoHa

LoHaは重みの更新を2つの低ランク行列のHadamard積（要素ごとの積）で表現します:

```
ΔW = (W1a × W1b) ⊙ (W2a × W2b)
```

ここで `W1a`, `W1b`, `W2a`, `W2b` はランク `network_dim` の低ランク行列です。LoHaは同じランクのLoRAと比較して学習可能なパラメータ数が **約2倍** になりますが、要素ごとの積により、より複雑な重み構造を捉えることができます。

### LoKr

LoKrはKronecker積を使って重みの更新を表現します:

```
ΔW = W1 ⊗ W2    （低ランクモードでは W2 = W2a × W2b）
```

元の重みの次元が因数分解されます（例: 512×512の重みが、W1が16×16、W2が32×32に分割されます）。W1は常にフル行列（小さい）で、W2はランク設定に応じて低ランク分解またはフル行列になります。LoKrは同じランクのLoRAと比較して **より小さいモデル** を生成する傾向があります。

</details>

## Inference / 推論

Trained LoHa/LoKr weights are saved in safetensors format, just like LoRA. The inference method depends on the architecture.

<details>
<summary>日本語</summary>

学習済みのLoHa/LoKrの重みは、LoRAと同様にsafetensors形式で保存されます。推論方法はアーキテクチャによって異なります。

</details>

### Architectures with built-in support / ネイティブサポートのあるアーキテクチャ

The following architectures automatically detect and load LoHa/LoKr weights without any additional options:

- Wan 2.1/2.2
- FramePack
- HunyuanVideo 1.5
- FLUX.2
- Qwen-Image series
- Z-Image

Use `--lora_weight` as usual:

```bash
python src/musubi_tuner/wan_generate_video.py ... --lora_weight path/to/loha_or_lokr.safetensors
```

<details>
<summary>日本語</summary>

以下のアーキテクチャでは、LoHa/LoKrの重みを追加オプションなしで自動検出して読み込みます:

- Wan 2.1/2.2
- FramePack
- HunyuanVideo 1.5
- FLUX.2
- Qwen-Image系
- Z-Image

通常通り `--lora_weight` を使用します:

```bash
python src/musubi_tuner/wan_generate_video.py ... --lora_weight path/to/loha_or_lokr.safetensors
```

</details>

### HunyuanVideo / FLUX.1 Kontext

For HunyuanVideo and FLUX.1 Kontext, the `--lycoris` option is required, and the [LyCORIS library](https://github.com/KohakuBlueleaf/LyCORIS) must be installed:

```bash
pip install lycoris-lora

python src/musubi_tuner/hv_generate_video.py ... --lora_weight path/to/loha_or_lokr.safetensors --lycoris
```

<details>
<summary>日本語</summary>

HunyuanVideoとFLUX.1 Kontextでは、`--lycoris` オプションが必要で、[LyCORIS ライブラリ](https://github.com/KohakuBlueleaf/LyCORIS)のインストールが必要です:

```bash
pip install lycoris-lora

python src/musubi_tuner/hv_generate_video.py ... --lora_weight path/to/loha_or_lokr.safetensors --lycoris
```

</details>

## Limitations / 制限事項

### LoRA+ is not supported / LoRA+は非対応

LoRA+ (`loraplus_lr_ratio` in `--network_args`) is **not supported** with LoHa/LoKr. LoRA+ works by applying different learning rates to the LoRA-A and LoRA-B matrices, which is specific to the standard LoRA architecture. LoHa and LoKr have different parameter structures and this optimization does not apply.

<details>
<summary>日本語</summary>

LoRA+（`--network_args` の `loraplus_lr_ratio`）はLoHa/LoKrでは **非対応** です。LoRA+はLoRA-AとLoRA-Bの行列に異なる学習率を適用する手法であり、標準的なLoRAのアーキテクチャに固有のものです。LoHaとLoKrはパラメータ構造が異なるため、この最適化は適用されません。

</details>

### Merging to base model / ベースモデルへのマージ

`merge_lora.py` currently supports standard LoRA only. LoHa/LoKr weights cannot be merged into the base model using this script.

For architectures with built-in LoHa/LoKr support (listed above), merging is performed automatically during model loading at inference time, so this limitation only affects offline merging workflows.

<details>
<summary>日本語</summary>

`merge_lora.py` は現在、標準LoRAのみをサポートしています。このスクリプトではLoHa/LoKrの重みをベースモデルにマージすることはできません。

LoHa/LoKrのネイティブサポートがあるアーキテクチャ（上記）では、推論時のモデル読み込み時にマージが自動的に行われるため、この制限はオフラインマージのワークフローにのみ影響します。

</details>

### Format conversion / フォーマット変換

`convert_lora.py` is extended to also support format conversion of LoHa/LoKr weights between Musubi Tuner format and Diffusers format for ComfyUI.

<details>
<summary>日本語</summary>

`convert_lora.py` は、LoRAに加えて、LoHa/LoKrの重みのフォーマット変換（Musubi Tuner形式とDiffusers形式間の変換）についてもサポートするよう、拡張されています。

</details>
