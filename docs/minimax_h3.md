# MiniMax-H3

This is the user guide: what to download, which training recipe fits your goal, and the commands for caching, training, sampling, and generation. Two companion documents cover the rest:

- `docs/minimax_h3_1f.md` — one-frame (image) generation and training: time indices, editing/inbetween datasets, reference-conditioned images.
- `docs/minimax_h3_advanced.md` — how the pieces work: timestep and loss internals, the guidance loss and teacher matching in depth, cache and quantization internals, generation internals, implementation provenance.

<details>
<summary>日本語</summary>

このドキュメントはユーザーガイドです。ダウンロードするファイル、目的に合った学習レシピの選び方、キャッシュ・学習・サンプル生成・推論のコマンドを説明します。残りは次の 2 文書にあります。

- `docs/minimax_h3_1f.md` — 1 フレーム（画像）の生成と学習。時間インデックス、編集・中割りデータセット、参照画像で条件付けする画像生成。
- `docs/minimax_h3_advanced.md` — 仕組みの解説。timestep と loss の内部、guidance loss と teacher matching の詳細、キャッシュと量子化の内部、生成の内部、実装の出典。

日本語の折り畳みはこのガイドのみに付けています。1f と advanced は英語のみです。表とコマンド例は英語部分を参照してください。

</details>

## Overview / 概要

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation, plus an experimental one-frame (image) mode for both.

The implementation follows the released MiniMax-H3 packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published full and pruned BF16 transformers, the full and pruned ConvRot INT8 transformers, and the ConvRot INT8 and NVFP4+AWQ Qwen3-VL text encoders.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

Thanks to [MiniMax AI](https://huggingface.co/MiniMaxAI) for releasing MiniMax-H3 as open weights.

<details>
<summary>日本語</summary>

Musubi Tuner は MiniMax-H3 の text-to-video-with-audio (T2VA)、first/last-frame-to-video-with-audio (FL2VA)、reference-to-video-with-audio (Ref2VA) の LoRA 学習と生成に対応しています。加えて、実験的な 1 フレーム（画像）モードを学習・生成の両方でサポートしています。

実装は公開された MiniMax-H3 の packing、Qwen3-VL による条件付け、video/audio 二重の flow スケジュール、2 種類の VAE レイアウトに従っています。公開されている full / pruned の BF16 transformer、full / pruned の ConvRot INT8 transformer、ConvRot INT8 および NVFP4+AWQ の Qwen3-VL テキストエンコーダーに対応します。

重みのダウンロード・使用の前に [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) を読み、同意してください。

MiniMax-H3 をオープンウェイトで公開してくださった [MiniMax AI](https://huggingface.co/MiniMaxAI) に感謝します。

</details>

## Model Files / モデルファイル

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | Supported file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| FL2VA and T2VA pruned transformer | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` |
| FL2VA and T2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| FL2VA and T2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Ref2VA pruned transformer | `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` |
| Ref2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` |
| Ref2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Qwen3-VL-32B ConvRot INT8 text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Qwen3-VL-32B NVFP4+AWQ text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

Thanks to the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team (Comfy-Org) for publishing the weights in these formats, including the pruned and quantized variants.

Which base for which task: T2VA and FL2VA (and every one-frame image recipe except reference-conditioned images) use the FL2VA transformer; Ref2VA uses the Ref2VA transformer. The pruned, ConvRot INT8, and NVFP4+AWQ files are drop-in replacements for their BF16 counterparts and are detected automatically from their tensor structure — pass them to `--dit` / `--text_encoder` and nothing else changes. What each one saves is summarized in [Memory and speed options](#memory-and-speed-options--メモリと速度のオプション). FP8 files and NVFP4 transformers are rejected.

The Qwen3-VL processor and config are downloaded by Transformers from the official [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repository (`processor` and `text_encoder` subfolders, a few config and tokenizer files only, no weights). The upstream `Qwen/Qwen3-VL-32B-Instruct` files are not interchangeable: the H3 tokenizer adds `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, and `<|caption_end|>` as special tokens, and the released prompt format writes dialogue and lyrics as `<d>[Language] ...</d>`.

<details>
<summary>日本語</summary>

上の表のファイルを [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) からダウンロードしてください。

pruned 版や量子化版を含め、これらの形式の重みを公開してくださった [ComfyUI](https://github.com/comfyanonymous/ComfyUI) チーム（Comfy-Org）に感謝します。

どのタスクにどの base を使うか: T2VA と FL2VA（および参照画像で条件付けする画像生成を除く、すべての 1 フレーム画像レシピ）は FL2VA transformer、Ref2VA は Ref2VA transformer を使います。pruned、ConvRot INT8、NVFP4+AWQ の各ファイルは BF16 版の置き換えとして使え、テンソル構造から自動判別されます。`--dit` / `--text_encoder` にそのまま渡すだけで、他に変更は不要です。それぞれの削減量は [Memory and speed options](#memory-and-speed-options--メモリと速度のオプション) にまとめています。FP8 ファイルと NVFP4 の transformer は受け付けません。

Qwen3-VL の processor と config は、Transformers が公式の [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) リポジトリ（`processor` と `text_encoder` サブフォルダの config と tokenizer のみ、重みは含まない）からダウンロードします。上流の `Qwen/Qwen3-VL-32B-Instruct` のファイルとは互換性がありません。H3 の tokenizer は `<d>`、`</d>`、`<|cutoff|>`、`<|lyrics_start|>`、`<|lyrics_end|>`、`<|caption_start|>`、`<|caption_end|>` を特殊トークンとして追加しており、公式のプロンプト形式ではセリフと歌詞を `<d>[Language] ...</d>` と書きます。

</details>

## Geometry And Media Contract / ジオメトリとメディアの規約

- Target video is 24 fps. Source videos are normalized to 24 fps from their frame timestamps, so `source_fps` is not needed and is ignored if set.
- Width and height must be positive multiples of 32.
- Frame count must be `17*n+5`. The released duration range is 5 to 15 seconds: at 24 fps, frame counts from 124 through 345 in steps of 17. `--allow_experimental_duration` bypasses only this duration check.
- Target audio is optional. When present it is decoded as stereo 32000 Hz audio; when absent, the cache stores a silence placeholder that is never used as a supervision target (see [Audio policy](#audio-policy)).
- Ref2VA references are ordered per record: from the JSONL `references` list for video datasets (the shared control-video fields are not used), and for image datasets also from control images (`control_directory` / `control_path`, one image reference per control). At most 12 references per record, of which at most 9 images, 3 videos, and 3 audio-bearing; at least one image or video; reference videos 2 to 15 seconds.
- Expanded Qwen conditioning is limited to 32768 rows. A BF16 text cache at the limit is approximately 320 MiB for one sample.

<details>
<summary>日本語</summary>

- 対象動画は 24 fps です。ソース動画はフレームのタイムスタンプから 24 fps に正規化されるので、`source_fps` は不要で、指定しても無視されます。
- 幅と高さは 32 の正の倍数である必要があります。
- フレーム数は `17*n+5` である必要があります。公開されている長さの範囲は 5〜15 秒で、24 fps では 124 から 345 まで 17 刻みです。`--allow_experimental_duration` はこの長さのチェックのみを外します。
- 対象音声は任意です。ある場合はステレオ 32000 Hz にデコードされ、ない場合はキャッシュに無音のプレースホルダが保存されます。プレースホルダは学習の教師には使われません（[Audio policy](#audio-policy) を参照）。
- Ref2VA の参照は各レコードで順序付きで定義されます（記述順に意味があります）。動画データセットでは JSONL の `references` リストから取ります（共通の control 動画フィールドは使いません）。画像データセットではこれに加えて control 画像（`control_directory` / `control_path`、control 1 枚が画像参照 1 つ）からも取れます。レコードあたり最大 12 参照、うち画像は最大 9、動画は最大 3、音声付きは最大 3。画像または動画が最低 1 つ必要で、参照動画は 2〜15 秒です。
- 展開後の Qwen 条件付けは 32768 行までです。上限での BF16 テキストキャッシュは 1 サンプルあたり約 320 MiB です。

</details>

## Choosing A Training Recipe / 学習レシピの選択

The released H3 checkpoints are CFG-distilled: they predict in an amplified "guided" space, and a LoRA trained on the plain flow-matching target pulls the model out of it. Video training then washes out and loses prompt adherence as it progresses; image training breaks structurally within about 50 steps. Plain flow training is therefore not offered as a recipe. Pick one of the three loss methods in Table A, then find your goal in Table B for the matching dataset shape, base, and flags.

### Table A: loss methods

| Method | What it does | Extra flags | Cost | Constraints |
| --- | --- | --- | --- | --- |
| Training adapter (de-distillation LoRA) | Merges a third-party (or Musubi-provided) adapter into the base at load time and trains with the plain flow loss on the de-distilled model. At inference the trained LoRA runs on the plain base, without the adapter. | `--base_weights adapter.safetensors` | none | BF16 source (with or without `--convrot_int8`); pre-quantized INT8 files cannot be merged into. Combining with the guidance loss or teacher matching is allowed but warned (see below). Training-time samples show the de-distilled model, not the plain base + LoRA. |
| Guidance loss | Re-anchors the flow target in the guided space using the model's own no-grad unconditional prediction. | text cache: `--uncond_output uncond.safetensors`; training: `--h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15 --h3_guidance_loss_uncond_cache uncond.safetensors` | +1 no-grad forward on ~85% of steps | any base, including INT8; not with teacher matching |
| Teacher matching | Trains a text-only student against the frozen base's prediction under privileged conditions (the clip's endpoints, the clip itself, or other pictures of the subject). | `--h3_teacher_matching --h3_teacher_conditions first,last` / `ref` / `subject_ref` plus the per-teacher settings in [Training](#training--学習) | +1 no-grad forward per step | student is always `--task t2va`; image targets: `subject_ref` only; not with the guidance loss |

Choosing between them: the adapter is the cheapest (no extra forward) and the least to configure; the guidance loss costs about 1.5x per step but needs no third-party file and works on a pre-quantized INT8 base; teacher matching is the recipe for identity training when the appearance is kept out of the captions (the teacher sees it, the student has to learn it). `--base_weights` also has ordinary uses (a style LoRA under a character LoRA), so combining it with the other two only logs a warning: the adapter authors advise against the guidance loss on top of it, and under teacher matching the merged base becomes the teacher.

<details>
<summary>日本語</summary>

公開されている H3 のチェックポイントは CFG 蒸留済みです。予測は増幅された「guided」空間で行われ、素の flow matching の target で LoRA を学習するとモデルがその空間から外れていきます。動画学習では進むにつれて色が抜け、プロンプトへの追従が落ちます。画像学習では 50 step ほどで構造が崩れます。そのため素の flow 学習はレシピとして提供していません。Table A の 3 つの loss 方式から 1 つを選び、Table B で目的に合う行を見つけて、データセットの形・base・フラグを決めてください。

**Table A の 3 方式:**

- **Training adapter（de-distillation LoRA）**: サードパーティ（または Musubi 提供）のアダプタをロード時に base へマージし、蒸留を解除したモデル上で素の flow loss で学習します。推論時は学習した LoRA をアダプタなしの素の base に適用します。フラグは `--base_weights adapter.safetensors`。追加コストはありません。制約: BF16 のソースが必要です（`--convrot_int8` の併用は可）。量子化済み INT8 ファイルにはマージできません。guidance loss や teacher matching との併用は可能ですが warning が出ます（後述）。学習中のサンプルは蒸留解除後のモデルの出力なので、素の base + LoRA の結果を表しません。
- **Guidance loss**: モデル自身の no-grad の無条件予測（CFG 推論時の uncond に相当）を使って、flow の target を guided 空間に置き直します。テキストキャッシュ時に `--uncond_output uncond.safetensors`、学習時に `--h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15 --h3_guidance_loss_uncond_cache uncond.safetensors` を指定します。コストは約 85% の step で no-grad forward が 1 回追加。制約: base は INT8 を含めて何でも可。teacher matching とは併用不可。
- **Teacher matching**: テキストのみの student を、H3 の条件入力（クリップの両端フレーム、クリップ自体、または同じ被写体の別の写真）を設定した frozen base の予測に合わせて学習します。これらの条件は student には与えません。`--h3_teacher_matching --h3_teacher_conditions first,last` / `ref` / `subject_ref` に、[Training](#training--学習) にある teacher ごとの設定を加えます。コストは毎 step no-grad forward が 1 回追加。制約: student は常に `--task t2va`。画像ターゲットでは `subject_ref` のみ。guidance loss とは併用不可。

**選び方**: アダプタは最も安く（forward の追加なし）設定も最少です。guidance loss は step あたり約 1.5 倍のコストですが、外部ファイルが不要で、量子化済み INT8 の base でも動きます。teacher matching は、外見をキャプションに書かずに identity を学習させたい場合のレシピです（teacher は外見を見ており、student はそれを学ぶ必要があります）。`--base_weights` にはキャラ LoRA の下に画風 LoRA を敷くといった通常の用途もあるので、他の 2 方式との併用は warning のみです。ただし、アダプタの作者は `--base_weights` によるアダプタ使用と guidance loss との併用を勧めていません。また teacher matching で `--base_weights` を指定すると、マージ後の base が teacher になります。

</details>

### Table B: recipes by goal

| Goal | Dataset shape | Base | Latent cache | Text cache | Training | Loss method |
| --- | --- | --- | --- | --- | --- | --- |
| Video: style, motion, general concept | `video_directory` or video JSONL | FL2VA | `--task t2va` | `--task t2va` (+`--uncond_output` for GL) | `--task t2va` | Adapter or guidance loss |
| Video: character identity, appearance kept out of captions | same | FL2VA | `--task fl2va` (endpoint teacher) or `--task t2va` (reference teacher) | `--task t2va --teacher_conditions first,last` or `ref` | `--task t2va` + [endpoint or reference teacher](#teacher-matching) | Teacher matching (`ref`: identity + voice; `first,last`: identity, base audio kept). Alternative: adapter or GL with a trigger word |
| Video: FL2VA (first/last-frame conditioned) | `video_directory` | FL2VA | `--task fl2va` | `--task fl2va` | `--task fl2va` | Adapter or GL |
| Video: Ref2VA (reference conditioned) | video JSONL with `references` | Ref2VA | `--task ref2va` | `--task ref2va` | `--task ref2va` | Adapter or GL |
| Image: plain image LoRA | `image_directory` or image JSONL | FL2VA | `--task t2va --one_frame` | `--task t2va --one_frame` | `--task t2va --one_frame --video_only` | Adapter or GL |
| Image: character identity, text-only at inference | image JSONL `references` or `control_directory` without `fp_1f_clean_indices` | FL2VA | `--task ref2va --one_frame` | `--task t2va --one_frame --teacher_conditions subject_ref` | `--task t2va --one_frame --video_only` + [subject-reference teacher](#teacher-matching) | Teacher matching (`subject_ref`). Alternative: plain image row with a trigger word |
| Image: editing / inbetween | image + timed controls (`fp_1f_clean_indices`, `fp_1f_target_index`) | FL2VA | `--task fl2va --one_frame` | `--task fl2va --one_frame` | `--task fl2va --one_frame --video_only` | Adapter or GL |
| Image: reference-conditioned at inference | image + untimed references | Ref2VA (FL2VA also works) | `--task ref2va --one_frame` | `--task ref2va --one_frame` | `--task ref2va --one_frame --video_only` | Adapter or GL |

Use the same `--task` for latent caching, text caching, and training unless the row says otherwise (the teacher-matching rows deliberately cache with a richer task than the student trains with). The image rows are described in detail in `docs/minimax_h3_1f.md`, including how the time indices and the timed-versus-untimed control distinction work. Mixed image+video training in one run is expected to work but is untested.

<details>
<summary>日本語</summary>

Table B の列は、目的 / データセットの形 / base / latent キャッシュのフラグ / テキストキャッシュのフラグ / 学習のフラグ / loss 方式です。行の目的は上から順に次のとおりです。

1. 動画: 画風・動き・一般的な概念（adapter または guidance loss）
2. 動画: キャラクターの identity、外見をキャプションに書かない（teacher matching。`ref` は identity と声、`first,last` は identity のみで base の音声を維持。代替はトリガーワード付きの adapter / GL）
3. 動画: FL2VA（両端フレーム条件付け。adapter または GL）
4. 動画: Ref2VA（参照条件付け。adapter または GL）
5. 画像: 通常の画像 LoRA（adapter または GL）
6. 画像: キャラクターの identity、推論時はテキストのみ（teacher matching の `subject_ref`。代替は 5 行目＋トリガーワード）
7. 画像: 編集・中割り（adapter または GL）
8. 画像: 推論時に参照画像で条件付け（adapter または GL）

行に別の指定がない限り、latent キャッシュ・テキストキャッシュ・学習で同じ `--task` を使ってください（teacher matching の行は、teacher に与える条件も含めてキャッシュするため、student の学習タスクより情報の多いタスクで意図的にキャッシュします）。画像の行の詳細（時間インデックス、時間付き control と時間なし参照の違い）は `docs/minimax_h3_1f.md` にあります。画像と動画を 1 回の学習で混ぜることは動作する見込みですが未検証です。

</details>

### Training adapters

Third-party de-distillation adapters that have been used with `--base_weights` on Musubi Tuner (all three load and train; output quality has not been evaluated here):

| Adapter | Target base | Rank | Notes |
| --- | --- | --- | --- |
| [circlestone-labs/MiniMax-H3-Image-Training-Adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter) | FL2VA, image-first (video and mixed "seem to work" per its README) | 64 | plain flow on 10k images / 10k steps; the README advises against the guidance loss on top of it |
| [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) v2 | FL2VA | 32 | ai-toolkit 0.13.4 |
| [ostris/minimax_h3_training_adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) `minimax_h3_ref2va_training_adapter_v1` | Ref2VA | 16 | ai-toolkit 0.12.23 |

All three are in the Diffusers key format (`diffusion_model.blocks.N....lora_A/lora_B.weight`, no alpha, so alpha = rank), which `--base_weights` and every generation `--lora_weight` route accept alongside Musubi's own format. A LoRA loaded from weights is applied to every module the file contains, including token refiner modules that Musubi's own training default leaves out.

Thanks to [circlestone-labs](https://huggingface.co/circlestone-labs) and [ostris](https://huggingface.co/ostris) for training and publishing these adapters.

<details>
<summary>日本語</summary>

Musubi Tuner で `--base_weights` に使えることを確認したサードパーティの de-distillation アダプタを表に示します（3 つともロードと学習は動作しますが、出力品質は評価していません）。表の列は、アダプタ / 対象の base / rank / 備考です。circlestone-labs のものは FL2VA base 向けで画像が主目的（README では動画・混在も「動くようだ」とされ、guidance loss との併用は非推奨）、ostris の v2 は FL2VA base 向け、ostris の `minimax_h3_ref2va_training_adapter_v1` は Ref2VA base 向けです。

3 つとも Diffusers のキー形式（`diffusion_model.blocks.N....lora_A/lora_B.weight`、alpha なし＝alpha は rank と同じ）です。`--base_weights` と生成側のすべての `--lora_weight` 経路は、Musubi 独自の形式に加えてこの形式を受け付けます。重みから読み込んだ LoRA はファイルに含まれるすべてのモジュールに適用されます。Musubi の学習デフォルトでは対象外の token refiner のモジュールも含まれます。

これらのアダプタを学習・公開してくださった [circlestone-labs](https://huggingface.co/circlestone-labs) と [ostris](https://huggingface.co/ostris) 氏に感謝します。

</details>

## Dataset Configuration / データセット設定

Dataset configuration uses the common TOML schema (`docs/dataset_config.md`). H3-specific rules: `batch_size` must be 1 in every H3 dataset (use gradient accumulation for a larger effective batch), and image and video datasets may share one TOML but must not share a `cache_directory`.

<details>
<summary>日本語</summary>

データセット設定は共通の TOML スキーマ（`docs/dataset_config.md`）を使います。H3 固有のルールは 2 つです。すべての H3 データセットで `batch_size` は 1 である必要があります（実効バッチを大きくするには gradient accumulation を使ってください）。画像データセットと動画データセットは 1 つの TOML に同居できますが、`cache_directory` は共有できません。

以下の各小節では、データの形ごとに設定例を示します。TOML と JSONL の例は英語部分を参照してください。

</details>

### Video directory (T2VA, FL2VA)

```toml
[general]
resolution = [768, 1344]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "/data/h3/videos"
cache_directory = "/data/h3/cache"
caption_extension = ".txt"
target_frames = [124]
frame_extraction = "head"
```

For a directory item such as `clip.mp4`, put the caption in `clip.txt`. FL2VA derives its first and last conditions from each selected target crop. Target audio is resolved in this order: exactly one same-stem audio sidecar such as `clip.wav`, then the video's embedded audio stream, then the silence placeholder.

<details>
<summary>日本語</summary>

ディレクトリ内の `clip.mp4` に対しては、キャプションを `clip.txt` に置きます。FL2VA は、選択された各ターゲットの切り出し範囲から最初と最後のフレームを条件として自動的に取り出します。対象音声は、同じベースファイル名で拡張子の異なる音声ファイル（`clip.wav` など。ちょうど 1 つであること）→ 動画に埋め込まれた音声トラック → 無音プレースホルダ、の順で解決されます。

</details>

### Video JSONL with references (Ref2VA)

```toml
[[datasets]]
video_jsonl_file = "/data/h3/ref2va.jsonl"
cache_directory = "/data/h3/cache-ref2va"
target_frames = [124]
frame_extraction = "head"
```

Each line holds the target plus its ordered references; relative paths resolve from the JSONL directory. A JSONL `audio_path` on the target takes precedence over sidecar and embedded audio.

```json
{"video_path":"targets/clip.mp4","audio_path":"targets/clip.wav","caption":"A singer performs under stage lights.","references":[{"type":"image","path":"refs/style.png"},{"type":"video","path":"refs/motion.mp4","audio_path":"refs/motion.wav"},{"type":"audio","path":"refs/voice.wav"}]}
```

A `video` reference uses its explicit `audio_path`, else its embedded track; `"audio_path": null` makes it a visual-only reference (motion or composition) even when the file has audio. A reference video without an audio track is likewise visual-only. `audio_path` is valid only on `video` references. The reference limits are listed under [Geometry And Media Contract](#geometry-and-media-contract--ジオメトリとメディアの規約). The same JSONL (with `references` on image targets) also feeds the subject-reference teacher for video identity training.

<details>
<summary>日本語</summary>

JSONL の各行はターゲットと順序付きの参照を持ちます（記述順に意味があります）。相対パスは JSONL のあるディレクトリから解決されます。ターゲットの `audio_path` は、同じベースファイル名の音声ファイルや埋め込み音声より優先されます。

`video` 参照は明示的な `audio_path` があればそれを、なければ埋め込みトラックを使います。`"audio_path": null` と書くと、ファイルに音声があっても映像のみの参照（動きや構図の参照）になります。音声トラックのない参照動画も同様に映像のみです。`audio_path` は `video` 参照にのみ指定できます。参照の上限は [Geometry And Media Contract](#geometry-and-media-contract--ジオメトリとメディアの規約) を参照してください。同じ JSONL（画像ターゲットに `references` を付けたもの）は、動画の identity 学習用の subject-reference teacher にも使います。

</details>

### Image directory or image JSONL (plain image LoRA)

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/data/h3/images"
cache_directory = "/data/h3/cache-images"
caption_extension = ".txt"
```

`image_jsonl_file` works as usual. Buckets snap to the 32-pixel grid. Since every image item carries silent audio rows, state the absence of sound in the caption (a `sound:`-style field describing a silent still) so the text stays consistent with what the model sees.

<details>
<summary>日本語</summary>

`image_jsonl_file` も通常どおり使えます。バケットは 32 ピクセルのグリッドに揃えられます。画像アイテムはすべて無音の audio 行を持つので、キャプションに音がないことを明記してください（`sound:` 形式のフィールドで無音の静止画であると書くなど）。モデルが見るものとテキストが一致します。

</details>

### Image with timed controls (editing / inbetween)

```toml
[[datasets]]
image_directory = "/data/h3/edit/targets"
control_directory = "/data/h3/edit/sources"
cache_directory = "/data/h3/cache-edit"
caption_extension = ".txt"
fp_1f_clean_indices = [0]     # control image positions (24 fps pixel-frame indices)
fp_1f_target_index = 24       # target position, required when controls are present
```

`fp_1f_clean_indices` is what makes the controls timed FL2VA anchors. Index choice matters a lot (a control at the target's own index trains against verbatim copying); see `docs/minimax_h3_1f.md`.

<details>
<summary>日本語</summary>

`fp_1f_clean_indices` を指定することで、control 画像が時間付きの FL2VA アンカーになります（`fp_1f_clean_indices` は control 画像の位置、`fp_1f_target_index` はターゲットの位置で、control がある場合は必須です。単位は 24 fps のピクセルフレームのインデックス）。インデックスの選び方は結果に大きく影響します（ターゲットと同じインデックスの control は、そのまま複写する挙動を無理やり上書きすることになります）。`docs/minimax_h3_1f.md` を参照してください。

</details>

### Image with untimed references (reference-conditioned images, subject-reference teacher)

Either per-record `references` in `image_jsonl_file` (same schema as the video JSONL, images and videos), or `control_directory` / `control_path` **without** `fp_1f_clean_indices`, in which case each control image becomes one image reference in index order:

```json
{"image_path": "/data/h3/char/targets/pose_01.png", "caption": "...", "references": [{"type": "image", "path": "refs/front.png"}]}
```

For the subject-reference teacher, the references should be *other* pictures of the same subject, and `caption` is the student's plain caption (a trigger word, appearance left out); an optional `teacher_caption` overrides the teacher's automatically wrapped caption.

<details>
<summary>日本語</summary>

参照の与え方は 2 通りです。`image_jsonl_file` のレコードごとの `references`（動画 JSONL と同じスキーマで、画像と動画が使えます）か、`fp_1f_clean_indices` を**指定しない** `control_directory` / `control_path` です。後者では各 control 画像がインデックス順に 1 つの画像参照になります。

subject-reference teacher に使う場合、参照は同じ被写体の*別の*写真にしてください。`caption` は student 用の素のキャプション（トリガーワードのみで外見は書かない）です。`teacher_caption` は省略可能で、省略時は student のキャプションに参照生成用のプロンプトが自動的に追加されます（`subject_definitions:` や `<Picture 1>` などの参照宣言）。明示的に `teacher_caption` を書くとこれを上書きできます。

</details>

## Caching / キャッシュ

Run latent caching and text-encoder caching once per dataset with the `--task` (and `--one_frame`) columns of Table B:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --cache_seed 42 \
  --skip_existing

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --text_cache_dtype bf16 \
  --skip_existing
```

- `--audio_vae` is always required: H3 always includes audio rows, even for silent items.
- `--skip_existing` rebuilds any cache whose stored metadata (task, cache seed, crop, format version, media and VAE fingerprints) no longer matches, so it is safe to leave on. Fingerprints are size + mtime, so a re-copied file triggers a one-time re-cache.
- The latent cache script prints the supervised-audio fraction at the end; a warning means no item had real audio.
- Text caching accepts the ConvRot INT8 and NVFP4+AWQ text encoders as well. On VRAM-limited GPUs add `--text_encoder_blocks_to_swap 50`, and `--text_encoder_attn_mode flash_attention_2` for long Ref2VA presentations.

Per-recipe additions to the text-caching command:

- **Guidance loss:** `--uncond_output /data/h3/uncond.safetensors` writes the tiny unconditional probe embedding (about 10 KB, one extra forward). `--uncond_text` overrides the probe text (default: a single space, which was selected as the true distillation uncond; see the advanced document).
- **Teacher matching:** `--teacher_conditions first,last`, `ref`, or `subject_ref` (always with `--task t2va`) stores the teacher's presentation next to the plain caption rows. The caption is shared; the teacher rows add the pictures or the reference declaration. The trainer hard-fails when the cache's teacher kind and `--h3_teacher_conditions` disagree, so re-cache text when switching teachers.

<details>
<summary>日本語</summary>

latent のキャッシュとテキストエンコーダー出力のキャッシュを、Table B の `--task`（と `--one_frame`）列の値でデータセットごとに 1 回ずつ実行します。コマンド例は英語部分を参照してください。

- `--audio_vae` は常に必要です。H3 は無音アイテムでも audio 行を持ちます。
- `--skip_existing` は、保存されたメタデータ（task、cache seed、切り出し位置、フォーマットのバージョン、メディアと VAE のフィンガープリント）が一致しないキャッシュを作り直すので、常に付けておいて安全です。フィンガープリントはサイズ＋mtime なので、ファイルをコピーし直すと 1 回だけ再キャッシュされます。
- latent キャッシュのスクリプトは終了時に、実音声のあるサンプルの割合を表示します。warning が出た場合、実音声のあるアイテムが 1 つもありません。
- テキストキャッシュは ConvRot INT8 と NVFP4+AWQ のテキストエンコーダーも受け付けます。VRAM が少ない GPU では `--text_encoder_blocks_to_swap 50` を、Ref2VA の参照が多くテキストエンコーダーへの入力が長くなる場合は `--text_encoder_attn_mode flash_attention_2` を追加してください。

テキストキャッシュのコマンドへのレシピ別の追加:

- **Guidance loss**: `--uncond_output /data/h3/uncond.safetensors` で無条件プローブの埋め込み（約 10 KB、forward 1 回追加）を書き出します。`--uncond_text` でプローブのテキストを変えられます（デフォルトは半角スペース 1 つで、蒸留時の真の uncond として選ばれたものです。advanced 文書を参照）。
- **Teacher matching**: `--teacher_conditions first,last` / `ref` / `subject_ref`（常に `--task t2va` と組み合わせる）で、素のキャプションと同じファイルに teacher 用のプレゼンテーションを保存します。デフォルトではキャプションは共有で、teacher の行には画像や参照宣言が加わります。キャッシュの teacher 種別と `--h3_teacher_conditions` が一致しないとトレーナーはエラーで停止するので、teacher を切り替えるときはテキストを再キャッシュしてください。

</details>

## Training / 学習

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --network_dim 16 \
  --network_alpha 16 \
  --sdpa \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --blocks_to_swap 48 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir /data/h3/output \
  --output_name h3-lora \
  <loss method flags>
```

`--network_module` defaults to `networks.lora_minimax_h3`, whose default targets are `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` in the 50 main DiT blocks. `--timestep_sampling uniform`, `--weighting_scheme none`, and `--discrete_flow_shift 1.0` are the H3 defaults and the only accepted values: H3 draws one base time per item and derives the video and audio sigmas from it with its own two shifts (12 and 3). `--min_timestep` / `--max_timestep` clip that base time (1000 = pure noise) before the shifts; the conversion to per-stream sigmas is tabulated in the advanced document.

Add exactly one of the following.

<details>
<summary>日本語</summary>

基本の学習コマンドは英語部分を参照してください。末尾の `<loss method flags>` の位置に、以下の小節のいずれか 1 つのフラグを追加します。

`--network_module` のデフォルトは `networks.lora_minimax_h3` で、デフォルトの対象は 50 個のメイン DiT ブロックの `attn.qkv_proj`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2` です。`--timestep_sampling uniform`、`--weighting_scheme none`、`--discrete_flow_shift 1.0` が H3 のデフォルトであり、これ以外の値は受け付けません。H3 はアイテムごとに base の時刻を 1 つ引き、そこから独自の 2 つの shift（12 と 3）で video と audio の sigma を導出します。`--min_timestep` / `--max_timestep` はこの base の時刻（1000 が純ノイズ）を shift の前に切り詰めます。各ストリームの sigma への換算表は advanced 文書にあります。

</details>

### Training adapter

```text
--base_weights /models/minimax_h3_training_adapter.safetensors
```

Works with a BF16 `--dit`, with or without `--convrot_int8` (the adapter is merged into the BF16 weights during the streaming load and quantized with them). Pre-quantized INT8 files are rejected. Use the trained LoRA on the plain base at inference; do not judge it by the training-time samples, which show the de-distilled model.

<details>
<summary>日本語</summary>

`--base_weights` にアダプタのファイルを指定します。BF16 の `--dit` で動作し、`--convrot_int8` の有無は問いません（アダプタはストリーミングロード中に BF16 の重みへマージされ、一緒に量子化されます）。量子化済みの INT8 ファイルは受け付けません。学習した LoRA は推論時に素の base に適用してください。学習中のサンプルは蒸留解除後のモデル、かつ CFG なしの出力なので、それで品質を判断しないでください。

</details>

### Guidance loss

```text
--h3_guidance_loss_scale 4.0 \
--h3_guidance_loss_sigma_min 0.15 \
--h3_guidance_loss_uncond_cache /data/h3/uncond.safetensors
```

Scale 3-4 (4 is more reliable for longer runs). `--h3_guidance_loss_sigma_min 0.15` skips the extra forward on the lowest-noise ~15% of steps, where the correction is mostly amplified noise; `0` applies it always. `--h3_guidance_loss_scale_audio` sets a separate audio scale. Each step logs `guidance/applied` and the gap magnitudes `guidance/video_gap_rms` / `guidance/audio_gap_rms`.

<details>
<summary>日本語</summary>

scale は 3〜4（長い学習では 4 のほうが安定します）。`--h3_guidance_loss_sigma_min 0.15` は、ノイズが最も少ない約 15% の step で追加の forward を省きます。この帯域では補正の大半がノイズの増幅にすぎません。`0` にすると常に適用します。`--h3_guidance_loss_scale_audio` で音声側の scale を別に指定できます。各 step で `guidance/applied` と、差分の大きさ `guidance/video_gap_rms` / `guidance/audio_gap_rms` がログに出ます。

</details>

### Teacher matching

The student is always `--task t2va`. Three teachers, each with its validated starting recipe:

**Endpoint teacher** (`first,last`; latent cache `--task fl2va`; identity from video, base audio behavior preserved):

```text
--h3_teacher_matching --h3_teacher_conditions first,last \
--h3_teacher_condition_sigma_max 0.75 --h3_teacher_loss_dc_weight 0.3 --h3_timestep_focus_prob 0.5
```

**Reference teacher** (`ref`; latent cache `--task t2va` or `fl2va`; identity and voice from video — the teacher copies each clip's actual audio, so the audio must be worth learning, otherwise lower `--audio_loss_weight` or pass `--video_only`):

```text
--h3_teacher_matching --h3_teacher_conditions ref \
--h3_teacher_condition_sigma_max 0.75 --h3_teacher_loss_dc_weight 0.3 --h3_timestep_focus_prob 0.5
```

**Subject-reference teacher** (`subject_ref`; latent cache `--task ref2va`; identity from other pictures of the subject; the only teacher for image targets, also usable for video):

```text
--h3_teacher_matching --h3_teacher_conditions subject_ref \
--h3_teacher_condition_sigma_min 0.15 --h3_teacher_loss_mag_weight 0.5 --h3_teacher_loss_dc_weight 0.3 \
--learning_rate 3e-4 --lr_warmup_steps 50 --max_train_steps 500
```

What the knobs mean, briefly: `--h3_teacher_condition_sigma_max` turns the highest-noise band into a base-preservation anchor so composition decisions are not overwritten (`0.75` for the endpoint and reference teachers; the subject-reference teacher keeps the default `1.0` because identity is decided at the top of the range, and the trainer warns when the value does not match the teacher's recipe); `--h3_teacher_condition_sigma_min` is the mirror gate at the low end; `--h3_teacher_loss_dc_weight` below 1 stops the dataset's palette from being learned as a style shift (keep `1.0` for style LoRAs); `--h3_teacher_loss_mag_weight` below 1 prioritizes direction over magnitude (a candidate for the reference teacher too, where the remaining distillation wedge is mostly a magnitude effect); `--h3_timestep_focus_prob P` lands a fraction P of the draws in `[--h3_timestep_focus_min, --h3_timestep_focus_max)` (default 0.4-0.8 in base units, where content is decided), which roughly doubles the band's convergence speed at 0.5; `--h3_teacher_preservation_weight` (default 1.0) strengthens the anchor for long runs. The loss does not converge to zero (the teacher knows things the text cannot), the teaching-band residual can plateau after a few hundred steps, and the strongest checkpoints tend to sit at or just after the plateau — save and evaluate intermediate checkpoints. The mechanism, the sigma-binned logs (`teacher/*`), how to read them, and the metadata keys are in the advanced document.

<details>
<summary>日本語</summary>

student は常に `--task t2va` です。teacher は 3 種類あり、それぞれ検証済みの初期レシピがあります（フラグは英語部分を参照）。

- **Endpoint teacher**（`first,last`。latent キャッシュは `--task fl2va`）: 動画から identity を学習し、base の音声の挙動を維持します。
- **Reference teacher**（`ref`。latent キャッシュは `--task t2va` または `fl2va`）: 動画から identity と声を学習します。teacher は各クリップの実際の音声を複写するので、音声が学習に値するものである必要があります。そうでなければ `--audio_loss_weight` を下げるか `--video_only` を付けてください。
- **Subject-reference teacher**（`subject_ref`。latent キャッシュは `--task ref2va`）: 同じ被写体の別の写真から identity を学習します。画像ターゲットで使える唯一の teacher で、動画にも使えます。

各ノブの意味を簡単に説明します。`--h3_teacher_condition_sigma_max` は最もノイズの多い帯域を base 維持のアンカーに変え、構図の決定が上書きされないようにします（endpoint と reference teacher では `0.75`。subject-reference teacher は identity が範囲の上端で決まるためデフォルトの `1.0` のままにします。teacher のレシピと値が合わないとトレーナーが warning を出します）。`--h3_teacher_condition_sigma_min` は低ノイズ側の対になるゲートです。`--h3_teacher_loss_dc_weight` を 1 未満にすると、データセットの色調が画風のずれとして学習されるのを防ぎます（画風 LoRA では `1.0` のままにします）。`--h3_teacher_loss_mag_weight` を 1 未満にすると大きさより方向を優先します（reference teacher でも候補です。sigma_max / sigma_min で指定した teacher を適用する sigma の範囲内に残る CFG 蒸留の効果は、ベクトルの方向ではなく主に大きさに現れるため、方向を優先することで CFG 蒸留を維持しやすくなります）。`--h3_timestep_focus_prob P` は、最初に引いた時刻を確率 P で `[--h3_timestep_focus_min, --h3_timestep_focus_max)`（デフォルト 0.4〜0.8、base 単位。内容が決まる帯域）に再マップします。残りの 1-P は一様分布のままなので、たとえば 0.5 では帯域内に入る確率が 50% + 50% × 帯域幅 0.4 = 約 70% になり、収束がおよそ 2 倍速くなります。`--h3_teacher_preservation_weight`（デフォルト 1.0）は長い学習でアンカーを強めます。アンカーとは、sigma_max より上（または sigma_min より下）の step で teacher が条件なしの base として予測し、student を base に引き戻す項のことです。

loss はゼロには収束しません（teacher はテキストから分からないことを知っているので、student が最善の予測をしても teacher には追い付けません）。教育帯域の残差は数百 step でプラトーに達することがあり、最も良い checkpoint はプラトーの時点かその直後にあることが多いので、途中の checkpoint を保存して評価してください。仕組み、sigma ごとに分けたログ（`teacher/*`）の読み方、メタデータのキーは advanced 文書にあります。

</details>

### Audio policy

Every sample contributes the video loss. A sample cached with real audio additionally contributes `--audio_loss_weight` (default 1.0) times the audio loss; items without real audio never contribute audio loss. `--video_only` disables audio supervision entirely (the model still attends to the audio latents as context). Because H3 is single-stream, a video-only LoRA modifies the weights the audio path uses too: treat audio from a fully video-only LoRA as unconstrained output. Image datasets should always pass `--video_only` (their audio rows are silence placeholders and would contribute nothing anyway).

<details>
<summary>日本語</summary>

すべてのサンプルが video loss に寄与します。実音声付きでキャッシュされたサンプルはさらに `--audio_loss_weight`（デフォルト 1.0）倍の audio loss に寄与し、実音声のないアイテムは audio loss に寄与しません。`--video_only` は音声の教師あり学習を完全に無効にします（モデルは audio latent をコンテキストとして参照し続けます）。H3 は single-stream なので、video のみの LoRA も音声経路が使う重みを変更します。完全に video のみで学習した LoRA の音声は、音声側の loss による制約がないため、元モデルからドリフトした出力になる可能性があります。画像データセットでは常に `--video_only` を付けてください（audio 行は無音プレースホルダで、いずれにせよ何も寄与しません）。

</details>

## Memory And Speed Options / メモリと速度のオプション

All options combine with each other and with every recipe. Sizes are transformer weight sizes unless noted.

| Option | Effect | Notes |
| --- | --- | --- |
| Pruned transformer (`*_pruned_*` files, or `--prune_adaln` on a full BF16 file) | ~66 → ~40 GB BF16, ~34 → ~21 GB INT8; each swapped block ~40% smaller, block-swap steps faster by the same fraction | detected automatically; `--prune_adaln` prunes at load time with slightly better reconstruction than the published files and combines with `--convrot_int8` |
| ConvRot INT8 transformer (`*_int8_convrot` files, or `--convrot_int8` on a BF16 file) | ~66 → ~34 GB; block-swap step time roughly halved (transfer-bound) | bit-identical to the published INT8 files; `--base_weights` needs the BF16 file + `--convrot_int8`; requires triton for the fused kernels (`triton-windows` on Windows), otherwise a slower dequantizing fallback with the same memory saving |
| `--blocks_to_swap N` (up to 48 of 50) | streams N blocks from CPU | `--gradient_checkpointing` recommended |
| `--block_swap_h2d_only` | faster block swap for frozen-base LoRA training (no device-to-host copies) | requires `--gradient_checkpointing`; see `docs/block_swap.md` |
| ConvRot INT8 text encoder | text encoder ~48 → ~25 GB | wherever `--text_encoder` is accepted |
| NVFP4+AWQ text encoder | text encoder ~48 → ~15 GB | inference-only artifact (the text encoder is always frozen); `--nvfp4_scaled_mm` opts into faster W4A4 matmuls on Blackwell GPUs with PyTorch 2.10+ |
| `--text_encoder_blocks_to_swap N` (up to 50) | streams N of the 50 Qwen3-VL layers from CPU; at 50 only embedding, vision tower, norms, and two one-layer buffers stay resident | requires CUDA; combines with the quantized encoders; add `--text_encoder_attn_mode flash_attention_2` for long Ref2VA presentations, where SDPA can fall back to an O(L^2) FP32 kernel |
| `--compile` (training and generation) | torch.compile on the 50 DiT blocks | with block swap or an INT8 base the Linears stay eager; each new latent shape recompiles (`--compile_dynamic true` for varying shapes) |

The reference teacher's forward carries the full reference video and audio tokens, so its teacher step is slower and needs more memory than the endpoint teacher's.

<details>
<summary>日本語</summary>

すべてのオプションは互いに、またすべてのレシピと組み合わせられます。表のサイズは特記がない限り transformer の重みのサイズです。

- **Pruned transformer**（`*_pruned_*` ファイル、または full の BF16 ファイルに `--prune_adaln`）: BF16 で約 66 → 約 40 GB、INT8 で約 34 → 約 21 GB。スワップされる各ブロックが約 40% 小さくなり、block swap の step も同じ割合で速くなります。自動判別されます。`--prune_adaln` はロード時に刈り込み、公開ファイルよりわずかに良い再構成精度で、`--convrot_int8` と併用できます。
- **ConvRot INT8 transformer**（`*_int8_convrot` ファイル、または BF16 ファイルに `--convrot_int8`）: 約 66 → 約 34 GB。block swap の step 時間はおよそ半分（転送律速のため）。公開 INT8 ファイルとビット一致します。`--base_weights` を使うには BF16 ファイル＋`--convrot_int8` が必要です。fused カーネルには triton（Windows では `triton-windows`）が必要で、ない場合はメモリ削減はそのままで、遅い逆量子化のフォールバックになります。
- **`--blocks_to_swap N`**（50 ブロック中最大 48）: N ブロックを CPU からストリーミングします。`--gradient_checkpointing` を推奨します。
- **`--block_swap_h2d_only`**: frozen base の LoRA 学習向けの高速な block swap（device→host のコピーなし）。`--gradient_checkpointing` が必要です。`docs/block_swap.md` を参照してください。
- **ConvRot INT8 テキストエンコーダー**: 約 48 → 約 25 GB。`--text_encoder` を受け付けるすべての場所で使えます。
- **NVFP4+AWQ テキストエンコーダー**: 約 48 → 約 15 GB。推論専用のアーティファクトです（テキストエンコーダーは常に frozen）。`--nvfp4_scaled_mm` で Blackwell 世代の GPU＋PyTorch 2.10 以上において高速な W4A4 matmul を有効にできます。
- **`--text_encoder_blocks_to_swap N`**（最大 50）: Qwen3-VL の 50 層のうち N 層を CPU からストリーミングします。50 では embedding、vision tower、norm、1 層分のバッファ 2 つだけが常駐します。CUDA が必要で、量子化エンコーダーと併用できます。長い Ref2VA のプレゼンテーションでは、SDPA が O(L^2) の FP32 カーネルにフォールバックすることがあるので `--text_encoder_attn_mode flash_attention_2` を追加してください。
- **`--compile`**（学習と生成）: 50 個の DiT ブロックに torch.compile を適用します。block swap または INT8 base では Linear は eager のままです。latent の形状が変わるたびに再コンパイルされます（形状が変わる場合は `--compile_dynamic true`）。

reference teacher の forward は参照動画と音声のトークン全体を含むので、teacher の step は endpoint teacher より遅く、メモリも多く必要です。

</details>

## Training-Time Samples / 学習中のサンプル生成

Add the sampling assets and the normal sampling schedule flags to the training command:

```text
--sample_prompts /data/h3/sample_prompts.txt \
--sample_every_n_epochs 1 \
--video_vae /models/minimax_h3_video_vae_fp16.safetensors \
--audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
--text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors
```

Samples are written as muxed MP4s under `OUTPUT_DIR/sample` (one-frame samples as PNG). The text encoder is loaded on the accelerator before the transformer to prepare every prompt once, so `--sample_prompts` needs room for it at that point: about 50 GB for the BF16 artifact, ~25 GB INT8, ~15 GB NVFP4; `--text_encoder_blocks_to_swap 50` removes most of that.

All entries use the training `--task`. A `.txt` prompt file holds one prompt per line with the same line options as generation ([Batch and interactive modes](#batch-and-interactive-modes)); lines starting with `#` are skipped:

```text
# T2VA
A singer performs under stage lights. --w 768 --h 1344 --f 124 --s 30 --d 42
# FL2VA: first and last frame (--i / --ei), or an ordered --ci list for one-frame samples
Official-format FL2VA caption... --w 768 --h 1344 --f 124 --s 30 --d 42 --i first.png --ei last.png
# Ref2VA: inline references (--ref, repeatable) or a record of a JSONL file (--rj)
A cat sings. --w 768 --h 1344 --f 124 --s 30 --d 42 --ref refs/cat.png --ref refs/dance.mp4;audio=refs/song.wav
```

Relative `--ref` and `--rj` paths resolve from the prompt file's directory. A `.json` prompt file takes the same requests as objects (`prompt`, `width`, `height`, `frame_count`, `sample_steps`, `seed`; `first_frame` / `last_frame`, `reference_jsonl` + optional `reference_index`, or a `ref` list). `--n`/`--l`/`--g` are rejected (no negative prompt or CFG). Geometry must be 32-pixel aligned; frame counts are rounded down to `17*n+5`, and `--h3_allow_experimental_sample_duration` permits samples shorter than 5 seconds.

Two caveats: samples under a merged `--base_weights` adapter show the de-distilled model, not the plain base + LoRA. And a LoRA trained toward a small equilibrium (teacher matching in particular) can look weaker in generation than in samples when it is merged into the BF16 base, because the merge rounds deltas below a BF16 mantissa step away; pass `--lora_runtime_attach` to generation to reproduce the training-time forward.

<details>
<summary>日本語</summary>

学習コマンドにサンプル生成用のアセット（`--sample_prompts`、video VAE、audio VAE、テキストエンコーダー）と通常のサンプル生成スケジュールのフラグを追加します（英語部分の例を参照）。

サンプルは `OUTPUT_DIR/sample` に音声付きの MP4 として書き出されます（1 フレームのサンプルは PNG）。テキストエンコーダーは transformer より先にアクセラレータに読み込まれ、すべてのプロンプトを一度に処理するので、その時点で VRAM にテキストエンコーダー分の空きが必要です。BF16 で約 50 GB、INT8 で約 25 GB、NVFP4 で約 15 GB です。`--text_encoder_blocks_to_swap 50` でその大半を不要にできます。

すべてのエントリは学習の `--task` を使います。`.txt` のプロンプトファイルは 1 行に 1 プロンプトで、生成と同じ行オプション（[Batch and interactive modes](#batch-and-interactive-modes)）が使えます。`#` で始まる行は無視されます。英語部分の例では、T2VA、FL2VA（`--i` / `--ei` で最初と最後のフレーム、1 フレームのサンプルでは順序付きの `--ci` リスト）、Ref2VA（`--ref` を複数回、または JSONL のレコードを `--rj`）の 3 行を示しています。

`--ref` と `--rj` の相対パスはプロンプトファイルのディレクトリから解決されます。`.json` のプロンプトファイルでは同じ要求をオブジェクトで書けます（`prompt`、`width`、`height`、`frame_count`、`sample_steps`、`seed`。加えて `first_frame` / `last_frame`、`reference_jsonl`＋省略可能な `reference_index`、または `ref` のリスト）。`--n` / `--l` / `--g` は受け付けません（ネガティブプロンプトと CFG はありません）。サイズは 32 ピクセル単位、フレーム数は `17*n+5` に切り下げられ、`--h3_allow_experimental_sample_duration` で 5 秒未満のサンプルを許可できます。

注意点が 2 つあります。`--base_weights` でアダプタをマージした状態のサンプルは蒸留解除後のモデルの出力で、素の base + LoRA の結果ではありません。また、小さな均衡点に向かって学習した LoRA（特に teacher matching）は、BF16 の base にマージすると BF16 の仮数の刻みより小さい差分が丸められて消えるため、サンプルより生成のほうが効きが弱く見えることがあります。生成時に `--lora_runtime_attach` を付けると学習時の forward を再現できます。

</details>

## Generation / 生成

T2VA with the FL2VA base:

```bash
python minimax_h3_generate_video.py \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --prompt "A singer performs under stage lights." \
  --video_size 1344 768 \
  --video_length 124 \
  --infer_steps 30 \
  --seed 42 \
  --blocks_to_swap 48 \
  --save_path output.mp4
```

`--video_size HEIGHT WIDTH` (multiples of 32), `--video_length` (pixel frames, `17*n+5`; `1` selects the one-frame image mode of `docs/minimax_h3_1f.md`), `--infer_steps N` (N model evaluations; the official 50-step default corresponds to `--infer_steps 49`). `--seed` is optional and logged when drawn. `--save_path` takes a file name or a directory (auto-named `<timestamp>_<seed>`); an existing file is never overwritten. `--output_type` can save the latents instead of or next to the video, or the frames as PNGs plus `audio.wav`.

Task inputs:

- **FL2VA:** `--task fl2va --first_frame first.png --last_frame last.png` with the FL2VA base. Either picture alone is also valid (I2VA / L2VA; use the matching official instruction line in the prompt). Condition images are scaled to cover the canvas and center-cropped, exactly as training fits controls to the bucket.
- **Ref2VA:** `--task ref2va` with the Ref2VA base and either `--reference_jsonl file.jsonl --reference_index 0` (the training JSONL schema; the target media only identifies the record) or inline references: `--ref refs/cat.png --ref "refs/dance.mp4;audio=refs/song.wav" --ref refs/bgm.mp3`. `--ref PATH[;type=image|video|audio][;audio=AUDIO_PATH]` is repeatable in reference order; the type is inferred from the extension when omitted. `--prompt` supplies (or overrides) the caption.
- **Text cache instead of the text encoder:** T2VA and Ref2VA accept `--text_cache` (a dataset text cache whose fingerprint matches the prompt and media); FL2VA does not.

Add a trained LoRA with:

```text
--lora_weight /data/h3/output/h3-lora.safetensors --lora_multiplier 1.0
```

Every route accepts Musubi's format and the Diffusers format written by ai-toolkit and diffusion-pipe. With a BF16 base the LoRA is merged once after loading (with `--convrot_int8`, merged before quantization); with a pre-quantized INT8 base it is attached as a runtime branch, so LoRA generation does not need the BF16 file. `--lora_runtime_attach` forces the runtime branch on any base (see the caveat under [Training-Time Samples](#training-time-samples--学習中のサンプル生成)).

Model loading dominates single-shot latency (and `--convrot_int8` requantizes at every start), so repeated generation should use the batch or interactive mode.

<details>
<summary>日本語</summary>

FL2VA base での T2VA 生成のコマンド例は英語部分を参照してください。

`--video_size 高さ 幅`（32 の倍数）、`--video_length`（ピクセルフレーム数、`17*n+5`。`1` を指定すると `docs/minimax_h3_1f.md` の 1 フレーム画像モードになります）、`--infer_steps N`（モデル評価 N 回。公式のデフォルト 50 step は `--infer_steps 49` に相当します）。`--seed` は省略可能で、乱数で決めた場合はログに出ます。`--save_path` はファイル名またはディレクトリ（`<timestamp>_<seed>` で自動命名）で、既存ファイルは上書きされません。`--output_type` で動画の代わりに、または動画に加えて latent を保存したり、フレームを PNG＋`audio.wav` として保存したりできます。

タスクごとの入力:

- **FL2VA**: FL2VA base で `--task fl2va --first_frame first.png --last_frame last.png`。どちらか片方だけでも有効です（I2VA / L2VA。プロンプトには対応する公式の指示行を使ってください）。条件画像はアスペクト比を維持したままキャンバスを覆うように拡大縮小してから、中央でクロップされます。学習で control をバケットに合わせるのと同じ処理です。
- **Ref2VA**: Ref2VA base で `--task ref2va` に、`--reference_jsonl file.jsonl --reference_index 0`（学習と同じ JSONL スキーマ。ターゲットのメディアはレコードの識別にだけ使われます）か、インラインの参照 `--ref refs/cat.png --ref "refs/dance.mp4;audio=refs/song.wav" --ref refs/bgm.mp3` を組み合わせます。`--ref PATH[;type=image|video|audio][;audio=AUDIO_PATH]` は繰り返し指定でき、指定順が参照の順序になります。type を省略すると拡張子から推定されます。キャプションは `--prompt` で与えます（JSONL のキャプションを上書きします）。
- **テキストエンコーダーの代わりにテキストキャッシュ**: T2VA と Ref2VA は `--text_cache`（プロンプトとメディアにフィンガープリントが一致するデータセットのテキストキャッシュ）を受け付けます。FL2VA は受け付けません。

学習した LoRA は `--lora_weight` と `--lora_multiplier` で追加します。すべての経路が Musubi の形式と、ai-toolkit や diffusion-pipe が書き出す Diffusers 形式を受け付けます。BF16 base ではロード後に一度マージされます（`--convrot_int8` を付けた場合は量子化の前にマージ）。量子化済み INT8 base では実行時に LoRA が動的に適用されるので、LoRA を使う生成に BF16 ファイルは不要です。`--lora_runtime_attach` はどの base でも実行時の動的適用を強制します（[Training-Time Samples](#training-time-samples--学習中のサンプル生成) の注意点を参照）。

1 回の生成ではモデルのロードが所要時間の大半を占めます（`--convrot_int8` は起動のたびに再量子化します）。繰り返し生成する場合はバッチモードか対話モードを使ってください。

</details>

### Batch and interactive modes

Both read prompt lines in the shared sample-prompt vocabulary; unspecified options inherit the command line, and a line starting with `--` re-runs the command-line prompt with new options:

```text
A singer performs under stage lights. --w 768 --h 1344 --f 124 --d 42 --s 30
```

| Line option | Maps to |
| --- | --- |
| `--w`, `--h` | `--video_size` (`--w` is the width, `--h` the height) |
| `--f` | `--video_length` (`--f 1` selects one-frame mode) |
| `--d` | `--seed` |
| `--s` | `--infer_steps` |
| `--fs`, `--fsa` | `--h3_shift_video`, `--h3_shift_audio` |
| `--ofps`, `--skb` | `--output_fps`, `--stretch_keep_bands` (temporal stretch, see the advanced document) |
| `--i`, `--ei` | `--first_frame`, `--last_frame` (end image) |
| `--ci` | `--condition_image` (one-frame FL2VA; repeatable, ordered; replaces the session-level list) |
| `--ref` | `--ref` (repeatable; replaces the session-level list) |
| `--of` | `--one_frame_inference` |
| `--o` | output filename inside the output directory |

`--from_file prompts.txt` runs every line in four phases (condition encoding, text encoding, sampling, decoding), loading each model family once; peak VRAM matches single-shot generation, and each sampled latent is saved before decoding so a crash never loses finished work. `--interactive` keeps the text encoder and transformer resident for a console session; on 24 GB and below combine a quantized transformer, a generous `--blocks_to_swap`, and `--text_encoder_blocks_to_swap 50`, and budget host RAM for both artifacts. `--latent_path FILE...` decodes saved latents with only the VAEs loaded. Details of the phases, naming, and text-conditioning cache are in the advanced document.

<details>
<summary>日本語</summary>

どちらのモードも、学習中のサンプル生成と共通の書式でプロンプト行を読みます。指定しなかったオプションはコマンドラインの値を引き継ぎ、`--` で始まる行はコマンドラインのプロンプトを新しいオプションで再実行します。行オプションとコマンドラインオプションの対応は英語部分の表を参照してください（`--w` が幅、`--h` が高さです）。

`--from_file prompts.txt` は全行を 4 段階（条件のエンコード、テキストのエンコード、サンプリング、デコード）で処理し、各モデルを 1 回だけロードします。ピーク VRAM は 1 回の生成と同じで、サンプリング済みの latent はデコード前に保存されるので、途中でクラッシュしても完了分は失われません。`--interactive` はコンソールセッションの間、テキストエンコーダーと transformer を常駐させます。24 GB 以下では量子化 transformer＋大きめの `--blocks_to_swap`＋`--text_encoder_blocks_to_swap 50` を組み合わせ、両方のアーティファクト分のホスト RAM を確保してください。`--latent_path FILE...` は VAE だけをロードして保存済みの latent をデコードします。段階の詳細、ファイル名の規則、テキスト条件のキャッシュは advanced 文書にあります。

</details>

## Limitations / 制限事項

- Released BF16 and ConvRot INT8 (each full or pruned) FL2VA/Ref2VA transformer bases only.
- BF16, ConvRot INT8, or NVFP4+AWQ Qwen3-VL text encoder only.
- No FP8 artifact loading, and no NVFP4 transformer loading.
- No CFG or negative prompt.
- Video datasets take Ref2VA references from JSONL only (no reference-directory convention); control images as references are an image-dataset feature.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
- Plain flow-matching training without one of the three loss methods is not a supported recipe (it runs, but de-distills the model).

<details>
<summary>日本語</summary>

- 対応する transformer base は、公開されている BF16 と ConvRot INT8（それぞれ full と pruned）の FL2VA / Ref2VA のみです。
- テキストエンコーダーは BF16、ConvRot INT8、NVFP4+AWQ の Qwen3-VL のみです。
- FP8 アーティファクトと NVFP4 の transformer は読み込めません。
- CFG とネガティブプロンプトはありません。
- 動画データセットの Ref2VA 参照は JSONL のみです（参照ディレクトリの規約はありません）。control 画像を参照にできるのは画像データセットの機能です。
- データセットの `batch_size` は 1 に固定です。実効バッチを大きくするには gradient accumulation を使ってください。
- パディング付きの複数サンプル packed レイアウトはありません。
- 3 つの loss 方式のいずれも使わない素の flow matching 学習はサポートするレシピではありません（動作はしますが、モデルの蒸留が解除されていきます）。

</details>
