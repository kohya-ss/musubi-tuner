# torch.compile Support

## Overview / 概要

This document describes the `torch.compile` optimization feature in Musubi Tuner. PyTorch's `torch.compile` is a just-in-time (JIT) compilation feature that can significantly improve training and inference performance by optimizing model execution.

For technical details and implementation specifics, please refer to [Pull Request #722](https://github.com/kohya-ss/musubi-tuner/pull/722).

Also, refer to the official PyTorch documentation: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile

Note: `torch.compile` may not work well in various situations. Please refer to the "Limitations and Known Issues" section below for details. If it does not work, please use the traditional method for training/inference.

<details>
<summary>日本語</summary>

このドキュメントでは、Musubi Tunerにおける`torch.compile`最適化機能について説明します。PyTorchの`torch.compile`は、モデルの実行を最適化することで学習と推論のパフォーマンスを大幅に向上させることができるジャストインタイム(JIT)コンパイル機能です。

技術的な詳細や実装の詳細については、[Pull Request #722](https://github.com/kohya-ss/musubi-tuner/pull/722)を参照してください。

PyTorchの公式ドキュメントも参照してください: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#introduction-to-torch-compile

※ `torch.compile`は様々な要因でうまく動作しない場合があります。詳細は以下の「制限事項と既知の問題」セクションを参照してください。また動作しない場合には従来の方法での学習/推論を行ってください。

</details>

### Prerequisites / 前提条件

- triton is required for `torch.compile` to work effectively. For Windows, see [triton-windows repository](https://github.com/woct0rdho/triton-windows) for installation instructions.
- MSVC compiler is required on Windows for `--compile_dynamic` option. Visual Studio 2022 with C++ development tools or Visual Studio Build Tools 2022 is recommended. See [Windows Requirements for `--compile_dynamic`](#windows-requirements-for---compile_dynamic--windowsでの---compile_dynamic-の要件).

<details>
<summary>日本語</summary>

- `torch.compile`を効果的に動作させるにはtritonが必要です。Windowsの場合、インストール手順については[triton-windowsリポジトリ](https://github.com/woct0rdho/triton-windows)を参照してください。
- Windowsで`--compile_dynamic`オプションを使用するにはMSVCコンパイラが必要です。Visual Studio 2022のC++開発ツールまたはVisual Studio Build Tools 2022の使用を推奨します。[`--compile_dynamic`のWindows要件](#windows-requirements-for---compile_dynamic--windowsでの---compile_dynamic-の要件)を参照してください。

</details>

### Performance Improvements / パフォーマンス向上

The performance gains vary depending on hardware and settings. Here are some examples:

**Qwen-Image, 1328×1328, BS1: RTX A6000, Power Limit 180W, Windows:**
- Default mode: ~10.5% faster
- max-autotune-no-cudagraphs: ~11.1% faster

**RTX PRO 6000 Blackwell, Power Limit 250W, Windows:**
- Default mode: ~18.8% faster
- max-autotune-no-cudagraphs: ~25.2% faster

<details>
<summary>日本語</summary>

パフォーマンス向上は、ハードウェアと設定によって異なります。以下は一例です:

**Qwen-Image, 1328×1328, BS1: RTX A6000, Power Limit 180W, Windows:**
- デフォルトモード: 約10.5%高速化
- max-autotune-no-cudagraphs: 約11.1%高速化

**RTX PRO 6000 Blackwell, Power Limit 250W, Windows:**
- デフォルトモード: 約18.8%高速化
- max-autotune-no-cudagraphs: 約25.2%高速化

</details>

## Supported Architectures / サポートされているアーキテクチャ

`torch.compile` is supported for both training and inference in the following architectures:

- HunyuanVideo
- Wan2.1/2.2
- FramePack
- FLUX.1 Kontext
- Qwen-Image / Qwen-Image-Edit / Qwen-Image-Edit-2509

<details>
<summary>日本語</summary>

以下のアーキテクチャで、学習と推論の両方において`torch.compile`がサポートされています:

- HunyuanVideo
- Wan2.1/2.2
- FramePack
- FLUX.1 Kontext
- Qwen-Image / Qwen-Image-Edit / Qwen-Image-Edit-2509

</details>

## Command Line Arguments / コマンドライン引数

### Basic Arguments / 基本的な引数

- `--compile`: Enable torch.compile optimization
- `--compile_backend`: Backend to use (default: `inductor`)
- `--compile_mode`: Compilation mode (default: `default` for training, `max-autotune-no-cudagraphs` for inference)
  - Choices: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
- `--compile_dynamic`: Enable dynamic shapes support (default is None, equivalent to `auto`) (Requires Visual Studio 2022 C++ compiler on Windows)
  - Choices: `true`, `false`, `auto`
- `--compile_fullgraph`: Enable fullgraph mode
- `--compile_cache_size_limit`: Set cache size limit (default: PyTorch default, typically 8-32, recommended: 32)

So far, it has been observed that setting `compile_mode` to `max-autotune` may not work in some cases.
Also, `compile_fullgraph` may not work depending on the architecture.

If `compile_dynamic` is not set to `true`, recompilation will occur each time the shape of the model input changes. This may result in longer training times for the first epoch, but subsequent epochs will be faster.

### Additional Performance Arguments / 追加のパフォーマンス引数

- `--cuda_allow_tf32`: Allow TF32 precision on Ampere or newer GPUs (improves performance)
- `--cuda_cudnn_benchmark`: Enable cuDNN benchmark mode (may improve performance)

<details>
<summary>日本語</summary>

### 基本的な引数

- `--compile`: torch.compile最適化を有効にする
- `--compile_backend`: 使用するバックエンド（デフォルト: `inductor`）
- `--compile_mode`: コンパイルモード（デフォルト: 学習時は`default`、推論時は`max-autotune-no-cudagraphs`）
  - 選択肢: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
- `--compile_dynamic`: 動的形状サポートを指定する（デフォルトは None で `auto` 相当）（Windows環境ではVisual Studio 2022のC++コンパイラが必要）
  - 選択肢: `true`, `false`, `auto`
- `--compile_fullgraph`: フルグラフモードを有効にする
- `--compile_cache_size_limit`: キャッシュサイズ制限を設定（デフォルト: PyTorchのデフォルト、通常8-32、推奨: 32）

これまでに確認したところ、`compile_mode`は`max-autotune`に設定すると動作しないケースがあるようです。
また、`compile_fullgraph`はアーキテクチャにより動作しない場合があります。

`compile_dynamic`で `true` を指定しない場合、モデルの入力の形状が変わるごとに再コンパイルが発生します。最初のエポックの学習時間が長くなる可能性がありますが、その後のエポックでは高速化されます。

### 追加のパフォーマンス引数

- `--cuda_allow_tf32`: Ampereまたはそれ以降のGPUでTF32精度を許可する（パフォーマンス向上）
- `--cuda_cudnn_benchmark`: cuDNNベンチマークモードを有効にする（パフォーマンスが向上する可能性がある）

</details>

## Usage Examples / 使用例

### Training / 学習

#### Basic Usage / 基本的な使い方

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
  --dit path/to/dit \
  --dataset_config path/to/config.toml \
  (... other args ...) \
  --compile \
  --compile_cache_size_limit 32
```

※ Windows Command Prompt users should use ^ at the end of lines.

#### Advanced Usage / 高度な使い方

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/hv_train_network.py \
  --dit path/to/dit \
  --dataset_config path/to/config.toml \
  (... other args ...) \
  --compile \
  --compile_mode max-autotune-no-cudagraphs \
  --compile_cache_size_limit 32 \
  --cuda_allow_tf32 \
  --cuda_cudnn_benchmark
```

<details>
<summary>日本語</summary>

### 学習

#### 基本的な使い方

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
  --dit path/to/dit \
  --dataset_config path/to/config.toml \
  (... その他の引数 ...) \
  --compile \
  --compile_cache_size_limit 32
```

※ Windowsでコマンドプロンプトを使用する場合、末尾は ^ を使用してください。

#### 高度な使い方

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/hv_train_network.py \
  --dit path/to/dit \
  --dataset_config path/to/config.toml \
  (... その他の引数 ...) \
  --compile \
  --compile_mode max-autotune-no-cudagraphs \
  --compile_cache_size_limit 32 \
  --cuda_allow_tf32 \
  --cuda_cudnn_benchmark
```

</details>

### Inference / 推論

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
  --dit path/to/dit \
  --vae path/to/vae \
  --text_encoder path/to/text_encoder \
  --prompt "A beautiful landscape" \
  --compile \
  --compile_mode max-autotune-no-cudagraphs
```

The existing `--compile_args` option is deprecated. It is still available for now but will be removed in the future. Please use the new individual arguments as shown in the example above.

<details>
<summary>日本語</summary>

### 推論

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
  --dit path/to/dit \
  --vae path/to/vae \
  --text_encoder path/to/text_encoder \
  --prompt "A beautiful landscape" \
  --compile \
  --compile_mode max-autotune-no-cudagraphs
```

既存の `--compile_args` オプションは非推奨となりました。現時点では使用可能ですが、将来的には削除される予定です。上の使用例のように、新しい個別の引数を使用してください。

</details>

## Limitations and Known Issues / 制限事項と既知の問題

### Incompatible Options and Constraints / 互換性のないオプションと制約

- **`--compile_fullgraph` and `--split_attn`**: These options cannot be used together. The `--split_attn` option uses dynamic control flow that is incompatible with fullgraph mode.
- **`--blocks_to_swap`**: When using block swapping, `torch.compile` automatically disables compilation for Linear layers in swapped blocks to avoid conflicts. This may limit performance improvements.

### Windows Requirements for `--compile_dynamic` / Windowsでの `--compile_dynamic` の要件

**IMPORTANT**: On Windows, using `--compile_dynamic` requires:

1. **Visual Studio 2022** with C++ development tools installed
2. Either:
    - Running the training/inference script from **"x64 Native Tools Command Prompt for VS 2022"**
    - Running the training/inference script after setting environment variables by executing `vcvars64.bat` located in the Visual Studio installation directory. For example: `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

If you encounter compilation errors when using `--compile_dynamic` on Windows, make sure you are running from the correct command prompt.



<details>
<summary>日本語</summary>

**互換性のないオプションと制約**

- **`--compile_fullgraph` と `--split_attn`**: これらのオプションは同時に使用できません。`--split_attn`オプションは動的な制御フローを使用しており、フルグラフモードと互換性がありません。
- **`--blocks_to_swap`**: ブロックスワッピングを使用する場合、`torch.compile`は衝突を避けるため、スワップされるブロック内のLinearレイヤーのコンパイルを自動的に無効にします。そのため、速度向上が制限される可能性があります。

**Windowsでの `--compile_dynamic` の要件**

**重要**: Windowsで`--compile_dynamic`を使用する場合、以下が必要です:

1. **Visual Studio 2022** とC++開発ツールのインストール
2. 以下のいずれか：
    - **"x64 Native Tools Command Prompt for VS 2022"** からのスクリプト実行
    - vcvars64.batを実行して環境変数を設定した後にスクリプトを実行：Visual Studioのインストールディレクトリにある`vcvars64.bat`を実行して環境変数を設定します。例: `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

Windowsで`--compile_dynamic`を使用してコンパイルエラーが発生する場合は、正しい手順でコマンドプロンプトから実行していることを確認してください。

PyTorchの以下の公式ドキュメントも参照してください: https://docs.pytorch.org/tutorials/unstable/inductor_windows.html#install-a-compiler

</details>

## Recommended Settings / 推奨設定

### For Training / 学習向け

```bash
--compile \
--compile_mode default \
--compile_cache_size_limit 32 \
--cuda_allow_tf32 \
--cuda_cudnn_benchmark
```

### For Inference / 推論向け

```bash
--compile \
--compile_mode max-autotune-no-cudagraphs \
--compile_cache_size_limit 32
```

<details>
<summary>日本語</summary>

### 学習向け

```bash
--compile \
--compile_mode default \
--compile_cache_size_limit 32 \
--cuda_allow_tf32 \
--cuda_cudnn_benchmark
```

### 推論向け

```bash
--compile \
--compile_mode max-autotune-no-cudagraphs \
--compile_cache_size_limit 32
```

</details>

## Compilation Modes / コンパイルモード

- **`default`**: Balanced compilation with good performance and reasonable compile times. Recommended for training.
- **`reduce-overhead`**: Reduces Python overhead, useful for small models or frequent small operations.
- **`max-autotune`**: Maximum optimization with longer compile times. May provide best performance but increases initial compilation time. May not work on some architectures.
- **`max-autotune-no-cudagraphs`**: Similar to max-autotune but without CUDA graphs. Recommended for inference as it provides good performance improvements with better compatibility.

<details>
<summary>日本語</summary>

- **`default`**: バランスの取れたコンパイルで、適切なパフォーマンスと合理的なコンパイル時間を提供します。学習に推奨されます。
- **`reduce-overhead`**: Pythonのオーバーヘッドを削減します。小さなモデルや頻繁な小規模操作に有用です。
- **`max-autotune`**: コンパイル時間は長くなりますが、最大限の最適化を行います。最高のパフォーマンスを提供する可能性がありますが、初期コンパイル時間が増加します。アーキテクチャによっては動作しない場合があります。
- **`max-autotune-no-cudagraphs`**: max-autotuneと似ていますが、CUDAグラフを使用しません。良好な互換性で優れたパフォーマンス向上を提供するため、推論に推奨されます。

</details>

## Troubleshooting / トラブルシューティング

### First Iteration is Slow

This is expected behavior. `torch.compile` performs JIT compilation on the first forward pass, which takes extra time. Subsequent iterations will be much faster.

### Out of Memory Errors

If you encounter out-of-memory errors when using `torch.compile`, try:
- Using a smaller `--compile_cache_size_limit` value
- Reducing batch size
- Using `--compile_mode default` instead of `max-autotune`

### Compilation Errors on Windows

If using `--compile_dynamic` on Windows and encountering compilation errors:
1. Ensure Visual Studio 2022 with C++ development tools is installed
2. Run the script from "x64 Native Tools Command Prompt for VS 2022"
3. If issues persist, try without `--compile_dynamic`

<details>
<summary>日本語</summary>

**最初のイテレーションが遅い**

これは予想される動作です。`torch.compile`は最初のforward passでJITコンパイルを実行するため、追加の時間がかかります。その後のイテレーションははるかに高速になります。

**メモリ不足エラー**

`torch.compile`を使用してメモリ不足エラーが発生する場合は、次を試してください:
- より小さな`--compile_cache_size_limit`値を使用する
- バッチサイズを減らす
- `max-autotune`の代わりに`--compile_mode default`を使用する

**Windowsでのコンパイルエラー**

Windowsで`--compile_dynamic`を使用してコンパイルエラーが発生する場合:
1. C++開発ツールを含むVisual Studio 2022がインストールされていることを確認する
2. "x64 Native Tools Command Prompt for VS 2022"からスクリプトを実行する
3. 問題が解決しない場合は、`--compile_dynamic`なしで試す

</details>

## Additional Resources / 追加リソース

- [PyTorch torch.compile documentation](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch Inductor Windows documentation](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
- [Pull Request #722](https://github.com/kohya-ss/musubi-tuner/pull/722) - Technical implementation details

<details>
<summary>日本語</summary>

- [PyTorch torch.compile ドキュメント](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch Inductor Windows ドキュメント](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
- [Pull Request #722](https://github.com/kohya-ss/musubi-tuner/pull/722) - 技術的な実装の詳細

</details>
