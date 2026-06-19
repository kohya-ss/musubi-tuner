# Block Swap (CPU Offloading) / ブロックスワップ（CPUオフロード）

## Overview / 概要

Block swap reduces VRAM usage by keeping only some of the model's transformer blocks on the GPU at a time and offloading the rest to CPU (main RAM), streaming each block to the GPU just before it is needed. This lets you train or run inference on models that would otherwise not fit in VRAM, at the cost of some speed and additional main RAM.

Enable it with `--blocks_to_swap N`, which offloads `N` blocks to CPU. A larger `N` saves more VRAM but uses more main RAM and may slow things down. The maximum value of `N` and the recommended VRAM/RAM amounts are architecture-specific; see each architecture's documentation (e.g. [HunyuanVideo](./hunyuan_video.md), [Wan](./wan.md), [Qwen-Image](./qwen_image.md)).

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

<details>
<summary>日本語</summary>

ブロックスワップは、モデルのtransformerブロックのうち一部のみをGPU上に置き、残りをCPU（メインRAM）にオフロードし、各ブロックを必要になる直前にGPUへ転送することで、VRAM使用量を削減します。これにより、本来VRAMに収まらないモデルでも学習・推論が可能になりますが、速度の低下と追加のメインRAMが必要になります。

`--blocks_to_swap N`で有効化し、`N`個のブロックをCPUにオフロードします。`N`を大きくするほどVRAMを節約できますが、メインRAMの使用量が増え、速度が低下する場合があります。`N`の最大値や推奨VRAM/RAM量はアーキテクチャごとに異なるため、各アーキテクチャのドキュメント（[HunyuanVideo](./hunyuan_video.md)、[Wan](./wan.md)、[Qwen-Image](./qwen_image.md)など）を参照してください。

（block swapのアイデアは2kpr氏の実装に基づくものです。2kpr氏にあらためて感謝します。）

</details>

## Command-Line Options / コマンドラインオプション

- `--blocks_to_swap N`: Number of blocks to offload to CPU. Default: none (disabled).
- `--use_pinned_memory_for_block_swap`: Use pinned (page-locked) host memory for the swapped weights. This can speed up CPU↔GPU transfer, but increases shared VRAM usage on Windows. Whether it helps depends on your system (available main RAM and VRAM); in some environments not specifying it is faster.
- `--block_swap_h2d_only`: Use H2D-only block swap (frozen-base / LoRA training only). See [H2D-only Block Swap](#h2d-only-block-swap--h2dのみのブロックスワップ).
- `--block_swap_ring_size N`: (used with `--block_swap_h2d_only`) Number of GPU ring buffers for streamed blocks. `2` (default) = double buffering (one block being computed on while the next is prefetched); `1` = minimal VRAM but no transfer/compute overlap.

<details>
<summary>日本語</summary>

- `--blocks_to_swap N`: CPUにオフロードするブロック数。デフォルトはなし（無効）。
- `--use_pinned_memory_for_block_swap`: スワップする重みにピン留め（ページロック）メモリを使用します。CPU↔GPU間の転送が高速化される場合がありますが、Windowsでは共有VRAM使用量が増加します。効果はシステム構成（利用可能なメインRAMとVRAM）に依存し、環境によっては指定しないほうが高速です。
- `--block_swap_h2d_only`: H2Dのみのブロックスワップを使用します（ベース凍結＝LoRA学習専用）。[H2Dのみのブロックスワップ](#h2d-only-block-swap--h2dのみのブロックスワップ)を参照してください。
- `--block_swap_ring_size N`:（`--block_swap_h2d_only`と併用）ストリーミング用GPUリングバッファの数。`2`（デフォルト）でダブルバッファ（1つのブロックを計算しつつ次をプリフェッチ）、`1`で最小VRAM（ただし転送と計算のオーバーラップなし）。

</details>

## H2D-only Block Swap / H2Dのみのブロックスワップ

H2D-only block swap (`--block_swap_h2d_only`) is an optimized mode for **frozen-base training** (LoRA / LoHa / LoKr, where the base model weights are not updated).

**※Currently this implementation is only available for CUDA-based architectures.**

The classic block swap *exchanges* a block: as a block is consumed it is copied back to the CPU (Device→Host, "D2H") while the next one is copied in (Host→Device, "H2D"). When the base weights are frozen, the CPU already holds an identical copy, so the D2H half is pure overhead. H2D-only keeps a permanent master copy of every streamed weight on the CPU and only ever copies Host→Device, removing the D2H transfer entirely. This can noticeably improve training throughput, and the benefit is largest with `--fp8_base` / `--fp8_scaled` (smaller weights mean less data to transfer).

With less transfer time, many models can achieve complete overlap of compute and transfer, and H2D-only can provide a significant speed boost. For example, in LoRA training of Qwen-Image on an RTX 3090 with a configuration that swaps 40 blocks, classic block swap runs at about 14 seconds/sample while H2D-only runs at about 11 seconds/sample, a roughly 25% speedup.

Note that on Max-Q devices and similar, power limits may cause H2D-only to run slower than no block swap at all, even with compute-transfer overlap.

**Requirements:**

- **Frozen base weights only** (LoRA / LoHa / LoKr). It cannot be used for full fine-tuning; the trainer stops with an error if a trainable base weight is detected.
- **`--gradient_checkpointing` is required for training.** Without it, training fails during the backward pass with an autograd error such as `one of the variables needed for gradient computation has been modified by an inplace operation ... version N; expected version M`. H2D-only streams weights into a small, reused GPU ring buffer; gradient checkpointing re-reads each weight at recompute time, which keeps autograd's version tracking consistent. The trainer checks this up front and reports an actionable message, so a missing `--gradient_checkpointing` is caught immediately rather than as the opaque error above.

**`--block_swap_ring_size`** controls how many GPU buffers stream the swapped blocks. `2` (default) overlaps the transfer of the next block with computation on the current one. `1` uses the least VRAM but removes the overlap; combined with `--fp8_base` this is a good minimal-memory configuration.

H2D-only is available for all architectures (training). Inference is not affected — see [Inference](#inference--推論).

<details>
<summary>日本語</summary>

H2Dのみのブロックスワップ（`--block_swap_h2d_only`）は、**ベース凍結の学習**（LoRA / LoHa / LoKr。ベースモデルの重みを更新しない学習）向けに最適化したモードです。

**※現在の実装はCUDAベースのアーキテクチャでのみ利用可能です。**

従来のブロックスワップはブロックを*入れ替え*ます。あるブロックを使い終わるとCPUへ書き戻し（Device→Host、「D2H」）、同時に次のブロックを転送（Host→Device、「H2D」）します。ベースの重みが凍結されている場合、CPU側には既に同一のコピーがあるため、D2Hは完全に無駄です。H2D-onlyは、ストリーミングする全重みの恒久的なマスターコピーをCPUに保持し、常にHost→Deviceのみコピーすることで、D2H転送を完全に排除します。これにより学習スループットが向上することがあり、その効果は`--fp8_base` / `--fp8_scaled`使用時に最も大きくなります（重みが小さくなり転送データ量が減るため）。

転送時間が減るため多くのモデルで計算と転送は完全にオーバーラップ可能となり、H2D-onlyは大幅なスループット向上をもたらすことがあります。たとえばRTX 3090におけるQwen-ImageのLoRA学習では、40ブロックをスワップする構成で、通常のブロックスワップは約14秒／サンプルですが、H2D-onlyは約11秒／サンプルと約25%の速度向上が見られます。

なお、計算と転送が完全にオーバーラップする場合でも、Max-Qデバイスなどでは電力制限により、ブロックスワップ未使用時よりも速度が低下することがあります。

**要件:**

- **ベース凍結の重みのみ**（LoRA / LoHa / LoKr）。フルファインチューニングでは使用できません。学習可能なベース重みが検出されると、エラーで停止します。
- **学習時は`--gradient_checkpointing`が必須です。** 指定しないと、バックワードパスで`one of the variables needed for gradient computation has been modified by an inplace operation ... version N; expected version M`のようなautogradエラーで失敗します。H2D-onlyは重みを小さな再利用GPUリングバッファへストリーミングしますが、gradient checkpointingは再計算時に各重みを読み直すため、autogradのversion管理が整合します。学習側で事前にチェックし分かりやすいメッセージを表示するので、`--gradient_checkpointing`の付け忘れは上記の難解なエラーではなく即座に検出されます。

**`--block_swap_ring_size`** は、スワップするブロックをストリーミングするGPUバッファの数を制御します。`2`（デフォルト）は次ブロックの転送を現ブロックの計算とオーバーラップさせます。`1`はVRAM使用量が最小ですがオーバーラップがなくなります。`--fp8_base`と組み合わせると最小メモリ構成として有効です。

H2D-onlyは全アーキテクチャ（学習）で利用できます。推論には影響しません（[推論](#inference--推論)を参照）。

</details>

## Inference / 推論

For inference, block swap uses the classic (exchange) implementation. `--block_swap_h2d_only` currently applies to training only; specifying it for inference does not change the offloading path. (Inference is forward-only with frozen weights, so H2D-only would be applicable in principle, but it is not wired into the inference scripts yet.)

<details>
<summary>日本語</summary>

推論では、ブロックスワップは従来の（入れ替え）実装を使用します。`--block_swap_h2d_only`は現状学習時のみ有効で、推論で指定してもオフロード経路は変わりません。（推論は重み凍結のforwardのみなので原理的にはH2D-onlyを適用可能ですが、推論スクリプトにはまだ組み込まれていません。）

</details>

## Notes / 補足

- **torch.compile:** Block swap is compatible with `--compile`, but when block swap is active the speed benefit of compilation is limited, because the swapped Linear layers are excluded from compilation. See [Using torch.compile](./torch_compile.md).
- **Main RAM:** Block swap holds the offloaded blocks in main RAM, and `--use_pinned_memory_for_block_swap` makes that memory page-locked. Larger `--blocks_to_swap` increases main RAM usage; see each architecture's documentation for recommended amounts.

<details>
<summary>日本語</summary>

- **torch.compile:** ブロックスワップは`--compile`と併用できますが、ブロックスワップが有効なときはコンパイルによる高速化の効果は限定的です（スワップ対象のLinear層がコンパイルから除外されるため）。[torch.compileの使用方法](./torch_compile.md)を参照してください。
- **メインRAM:** ブロックスワップはオフロードしたブロックをメインRAMに保持し、`--use_pinned_memory_for_block_swap`を指定するとそのメモリがページロックされます。`--blocks_to_swap`を大きくするとメインRAM使用量が増えます。推奨量は各アーキテクチャのドキュメントを参照してください。

</details>
