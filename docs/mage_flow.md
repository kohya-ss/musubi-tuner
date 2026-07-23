# Mage-Flow

Musubi Tuner has experimental LoRA training and inference support for
[Microsoft Mage-Flow](https://microsoft.github.io/Mage/flow/) and Mage-Flow-Edit.
The model implementation is pinned to upstream Mage commit
`ea7109b3515ddd995c2e1212656dc1bc3a9607b7`.

This integration has passed synthetic contract and tiny-model tests. It has not
yet passed the real released-weight parity checklist in this repository, so it
is not a general-availability compatibility claim.

## Supported Scope

- Mage-Flow T2I and Mage-Flow-Edit with one to three ordered references
- LoRA training only
- BF16, scaled FP8, gradient checkpointing, block swap and `torch.compile`
- PyTorch SDPA by default and optional FlashAttention 2
- fixed released DiT dimensions and Qwen3-VL-4B conditioning
- MageVAE image latents with 128 channels and 16x spatial downsampling
- current bucket training with a 16-pixel image step

Not supported:

- full-model fine-tuning
- public native-resolution packing
- checkpoint directories or sharded component paths
- inferred T2I/Edit or Base/aligned/Turbo identity from filenames
- SageAttention, xFormers, FlashAttention 3/4
- official content screening or Gaussian-Shading watermarking
- unverified Comfy-Org checkpoint layouts

## Component Files

Pass one regular `.safetensors` file for each component:

```text
--dit            Mage-Flow DiT
--vae            combined MageVAE encoder and decoder
--text_encoder   Qwen3-VL-4B backbone
```

Directories and shard lists are rejected. The loaders inspect every key, shape
and dtype before allocating the released model and then load strictly. The Qwen
language-model head is not needed because conditioning uses only the final
backbone hidden state.

The current loader recognizes the pinned Microsoft key layouts documented by
the command errors. It does not guess a layout from a filename or tensor shape.

## Dataset

Use the regular image dataset configuration. Target images and captions are the
same as for other image architectures. For Edit, add `control_directory`; the
target is the desired edited image and controls are the source references.

Multiple controls use the existing indexed naming:

```text
target/image1.png
target/image1.txt
controls/image1_0.png
controls/image1_1.png
controls/image1_2.png
```

Indices are semantic order and must be contiguous from zero. Edit accepts one,
two or three controls. See [Dataset Configuration](./dataset_config.md#image-dataset-with-control-images).

### Bucket Training Versus Native Packing

The first release deliberately keeps Musubi's public bucket pipeline. A batch
therefore contains compatible target sizes; Edit also separates batches by
reference count and ordered reference shapes. This preserves existing collator
behavior and predictable memory use.

Inside the Mage model, every batch is converted to the official-style packed
contract:

```text
image_tokens + image_cu_seqlens
text_tokens  + text_cu_seqlens
image_shapes + target_token_mask
```

This internal ABI already accepts heterogeneous segment lengths and prevents
cross-sample attention. It is not the same as native-resolution packing:
native packing would form a batch directly from unrelated resolutions according
to a token budget, without first grouping samples into equal-resolution
buckets. That future scheduler can use the retained contract, but this release
does not add native scheduling or change common collator semantics. It only
routes the new Edit identity through the existing multi-control discovery path
and validates Mage cache metadata; existing architectures keep their behavior.

## Cache

Use the same mode for both cache passes. Omit `--is_edit` for T2I.

```bash
python mage_flow_cache_latents.py \
  --dataset_config dataset.toml \
  --vae path/to/mage_vae.safetensors \
  --vae_dtype bfloat16 \
  --batch_size 1

python mage_flow_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --processor Qwen/Qwen3-VL-4B-Instruct \
  --text_encoder_dtype bfloat16 \
  --batch_size 1
```

For Edit, add `--is_edit` to both commands:

```bash
python mage_flow_cache_latents.py \
  --is_edit \
  --dataset_config edit_dataset.toml \
  --vae path/to/mage_vae.safetensors \
  --vae_dtype bfloat16

python mage_flow_cache_text_encoder_outputs.py \
  --is_edit \
  --dataset_config edit_dataset.toml \
  --text_encoder path/to/qwen3_vl_4b.safetensors
```

MageVAE posterior samples are derived from the architecture, stable item key,
target/control role and `--seed`. Reordering items or changing cache batch size
therefore does not change an item's cached latent.

## Train

The trainer defaults to the fixed Mage LoRA module, BF16, shifted flow matching
with shift 6, and no loss weighting:

```bash
accelerate launch mage_flow_train_network.py \
  --dataset_config dataset.toml \
  --dit path/to/mage_flow_dit.safetensors \
  --network_dim 32 \
  --network_alpha 32 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --output_dir output \
  --output_name mage_flow_lora \
  --sdpa
```

Add `--is_edit` for Mage-Flow-Edit. The target follows
`x_t = (1-t)z + t*epsilon`, and the velocity target is `epsilon-z`. Edit
references remain clean, share the sample timestep modulation, and are excluded
from loss.

LoRA is limited to attention and image/text feed-forward linear layers inside
the 12 repeated transformer blocks. Modulation, normalization, input, timestep
and output projections are excluded. User include/exclude patterns are ignored
with a warning; they cannot narrow or expand this fixed supported scope.

Useful memory options:

```text
--gradient_checkpointing
--blocks_to_swap 0..10
--block_swap_h2d_only --gradient_checkpointing
--fp8_base --fp8_scaled
--compile
```

Plain `--fp8_base` is rejected. Scaled FP8 quantizes only supported repeated
block attention/MLP weights; modulation, norms and global projections stay in
the compute dtype. Eligible frozen base weights are converted while the DiT is
loaded, before compute-dtype LoRA modules are attached.

Use `--flash_attn` instead of `--sdpa` for optional FlashAttention 2. The
package remains an optional user installation.

## Generate

T2I Base-oriented defaults are 30 steps and CFG 5:

```bash
python mage_flow_generate_image.py \
  --dit path/to/mage_flow_dit.safetensors \
  --vae path/to/mage_vae.safetensors \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --prompt "a glass greenhouse above a quiet city" \
  --negative_prompt " " \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --cfg_scale 5 \
  --seed 42 \
  --output output.png
```

For Edit, repeat `--control_image` in semantic order:

```bash
python mage_flow_generate_image.py \
  --is_edit \
  --dit path/to/mage_flow_edit_dit.safetensors \
  --vae path/to/mage_vae.safetensors \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --control_image source.png \
  --control_image style_reference.png \
  --prompt "replace the sky with a late afternoon sky" \
  --max_size 1024 \
  --steps 30 \
  --cfg_scale 5 \
  --seed 42 \
  --output edit.png
```

Edit size precedence is explicit width plus height, then `--max_size` applied to
the primary reference aspect ratio, then the primary source size. Dimensions
are rounded down to multiples of 16. Qwen visual conditioning caps each
reference's long edge at 384 pixels; MageVAE receives references resized to the
output size.

Load adapters with `--lora_weight one.safetensors two.safetensors` and optional
matching `--lora_multiplier`. T2I/Edit architecture metadata must match the
selected mode. Cross-mode loading requires
`--allow_mage_architecture_mismatch`.

Generation selects attention with `--attn_mode sdpa` (default) or
`--attn_mode flash2`. Training uses `--sdpa` or the optional `--flash_attn`
flag instead.

Recommended explicit schedules:

| Family | Steps | CFG |
|---|---:|---:|
| Base | 30 | 5.0 |
| aligned | 20 (Edit: 30) | 5.0 |
| Turbo | 4 | 1.0 |

## Real-Weight Validation

Before treating a component set as compatible, verify:

1. each of DiT, MageVAE and Qwen is one readable safetensors file;
2. strict header and state-dict loading succeeds without ignored keys;
3. T2I conditioning matches the pinned official output after the 34-token drop;
4. Edit conditioning matches after the 64-token drop with one and three refs;
5. fixed packed DiT inputs match official BF16 outputs;
6. one-step and short Euler outputs match before watermark insertion;
7. one LoRA step, save/reload, T2I sample and one/three-reference Edit sample succeed;
8. SDPA, optional FA2, scaled FP8, checkpointing, block swap and compile are
   checked on the intended GPU.

Until that checklist is run with released files, this support remains
experimental.

## 日本語

Musubi Tuner は Microsoft Mage-Flow と Mage-Flow-Edit の LoRA 学習および
推論に実験的に対応しています。実装は upstream Mage の
`ea7109b3515ddd995c2e1212656dc1bc3a9607b7` に固定されています。

合成データによる契約テストと小型モデルテストは通過していますが、公開済み
実重みを使った数値 parity はまだ完了していません。そのため、現時点では
一般利用向けの完全互換を保証しません。

### 対応範囲

- Mage-Flow T2I
- 参照画像を順序付きで 1～3 枚使う Mage-Flow-Edit
- LoRA 学習のみ
- BF16、scaled FP8、gradient checkpointing、block swap、`torch.compile`
- 既定の PyTorch SDPA と、任意導入の FlashAttention 2
- 固定された公開 DiT 構成と Qwen3-VL-4B conditioning
- 128 channel、空間 16 倍圧縮の MageVAE latent
- 画像 step 16 の既存 bucket 学習

full-model fine-tuning、公開データローダーでの native-resolution packing、
directory/shard 形式の component、ファイル名からのモデル種別推測、
SageAttention、xFormers、FlashAttention 3/4、未確認の Comfy-Org key layout
には対応していません。

### コンポーネント

各 component には通常の `.safetensors` ファイルを 1 個ずつ指定します。

```text
--dit            Mage-Flow DiT
--vae            encoder と decoder を含む MageVAE
--text_encoder   Qwen3-VL-4B backbone
```

loader は大きなモデルを確保する前に全 key、shape、dtype を検証し、その後
strict load します。conditioning は最終 backbone hidden state のみを使うため、
Qwen の `lm_head.weight` は不要です。

### データセットと bucket

通常の image dataset 設定を使います。Edit では target を編集後の正解画像、
`control_directory` 内の画像を参照画像として扱います。

```text
target/image1.png
target/image1.txt
controls/image1_0.png
controls/image1_1.png
controls/image1_2.png
```

index は意味上の順序で、0 から欠番なく並べる必要があります。Edit は 1～3 枚
を受け付けます。

初版は既存の bucket pipeline を維持します。同じ batch の target size は互換
であり、Edit は参照枚数と各参照 shape でも batch を分けます。モデル内部では
次の packed ABI に変換されます。

```text
image_tokens + image_cu_seqlens
text_tokens  + text_cu_seqlens
image_shapes + target_token_mask
```

この ABI 自体は異なる segment 長を処理できますが、native-resolution packing
ではありません。native packing は、解像度 bucket を先に選ばず、異なる解像度
の sample を token budget まで直接詰めます。将来その scheduler を追加できる
契約は残していますが、初版では native scheduler や共通 collator の意味を変更
しません。新しい Edit identity を既存の複数 control 読み込みへ接続し、Mage
cache metadata を検証するだけで、既存 architecture の挙動は変えません。

### キャッシュ

T2I では次を実行します。

```bash
python mage_flow_cache_latents.py \
  --dataset_config dataset.toml \
  --vae path/to/mage_vae.safetensors \
  --vae_dtype bfloat16 \
  --batch_size 1

python mage_flow_cache_text_encoder_outputs.py \
  --dataset_config dataset.toml \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --processor Qwen/Qwen3-VL-4B-Instruct \
  --text_encoder_dtype bfloat16 \
  --batch_size 1
```

Edit では両方のコマンドに `--is_edit` を付けます。latent と text cache は
T2I/Edit で別の architecture identity を持ち、学習開始前に metadata を検証
します。MageVAE posterior の乱数は architecture、安定した item key、
target/control の役割、`--seed` から決まるため、item 順や cache batch size を
変えても同じ item の latent は変化しません。

### 学習

既定値は固定 Mage LoRA module、BF16、shift 6 の flow matching、loss weighting
なしです。

```bash
accelerate launch mage_flow_train_network.py \
  --dataset_config dataset.toml \
  --dit path/to/mage_flow_dit.safetensors \
  --network_dim 32 \
  --network_alpha 32 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --output_dir output \
  --output_name mage_flow_lora \
  --sdpa
```

Edit では `--is_edit` を追加します。target は
`x_t = (1-t)z + t*epsilon`、velocity target は `epsilon-z` です。参照 latent
には noise を加えませんが、target と同じ sample-level timestep modulation を
使います。loss は target token のみに適用されます。

LoRA scope は 12 個の repeated block 内の attention と image/text FFN linear
に固定されています。modulation、normalization、input、timestep、output
projection は対象外です。`include_patterns` / `exclude_patterns` を指定しても
警告を出して無視し、scope を狭めたり広げたりしません。

主な省メモリ option:

```text
--gradient_checkpointing
--blocks_to_swap 0..10
--block_swap_h2d_only --gradient_checkpointing
--fp8_base --fp8_scaled
--compile
```

unscaled の `--fp8_base` は拒否します。scaled FP8 変換は DiT load 中に対象の
凍結 base weight だけへ適用され、その後 compute dtype の LoRA を付加します。
学習 attention は `--sdpa`、または任意導入の FlashAttention 2 を使う
`--flash_attn` から選びます。

### 生成

T2I Base 向けの既定は 30 step、CFG 5 です。

```bash
python mage_flow_generate_image.py \
  --dit path/to/mage_flow_dit.safetensors \
  --vae path/to/mage_vae.safetensors \
  --text_encoder path/to/qwen3_vl_4b.safetensors \
  --prompt "a glass greenhouse above a quiet city" \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --cfg_scale 5 \
  --seed 42 \
  --output output.png
```

Edit では `--is_edit` を付け、意味上の順序で `--control_image` を 1～3 回指定
します。output size の優先順は、明示した `--width` と `--height`、
primary reference の縦横比に適用する `--max_size`、primary reference の元
size です。最終 size は 16 の倍数へ切り下げます。

推論 attention は `--attn_mode sdpa`（既定）または
`--attn_mode flash2` で選びます。LoRA は `--lora_weight` と、必要に応じて
`--lora_multiplier` で読み込みます。T2I/Edit metadata が現在の mode と異なる
場合は拒否し、意図的な cross-mode load のみ
`--allow_mage_architecture_mismatch` で許可します。

推奨 schedule:

| family | step | CFG |
|---|---:|---:|
| Base | 30 | 5.0 |
| aligned | 20（Edit は 30） | 5.0 |
| Turbo | 4 | 1.0 |

### 実重みでの検証

一般利用向けと判断する前に、次を公開 weight で確認してください。

1. DiT、MageVAE、Qwen がそれぞれ読み取り可能な単一 safetensors であること
2. 無視された key なしで header 検証と strict load が成功すること
3. T2I の Qwen 出力が 34 token 除去後に pinned upstream と一致すること
4. 参照 1 枚と 3 枚の Edit 出力が 64 token 除去後に一致すること
5. 固定 packed input の DiT BF16 出力が一致すること
6. watermark 前の 1-step および短い Euler 出力が一致すること
7. LoRA 1 step、保存/再読込、T2I、参照 1 枚/3 枚 Edit が成功すること
8. 使用予定 GPU で SDPA、任意の FA2、scaled FP8、checkpointing、block swap、
   compile を確認すること

この checklist が公開 weight で完了するまでは experimental support です。
