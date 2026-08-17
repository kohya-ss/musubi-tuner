> 📝 Click on the language section to expand / 言語をクリックして展開

# Static Watermark Masking / 静的ウォーターマークのマスク

Training footage often carries a burned-in logo or watermark. Without a mask, the model learns the watermark along with the concept, and it shows up in generated videos. Cropping the watermark away throws the rest of the frame away with it, and changes the composition the model sees.

This feature excludes the watermark region from the **loss** instead. The frame stays full size and is encoded normally; the masked pixels simply produce no gradients.

Only **static** watermarks are supported: the overlay must sit at the same place whenever it is on screen. Within that, it can be opaque or semi-transparent, and it does not have to be on screen the whole time — a watermark that briefly disappears, is occluded by bright content, or fades in and out is still detected, up to a configurable fraction of frames (`--frame_tolerance`, 10% by default). What a single mask cannot describe is a watermark that *moves* across the frame; drop those clips from the dataset instead.

The mask is entirely optional. Clips without a mask file are trained on in full, exactly as before.

<details>
<summary>日本語</summary>

学習用の動画には、ロゴやウォーターマークが焼き込まれていることがよくあります。マスクを使用しない場合、モデルは学習したい概念と一緒にウォーターマークも学習してしまい、生成される動画にウォーターマークが現れます。ウォーターマークを切り取ると、フレームの他の部分も一緒に失われ、モデルが見る構図も変わってしまいます。

この機能では、代わりにウォーターマークの領域を**損失（loss）**から除外します。フレームはフルサイズのまま通常どおりエンコードされ、マスクされたピクセルからは勾配が生じません。

対応しているのは**静的な**ウォーターマークのみです。オーバーレイは、画面に表示されている間は常に同じ位置・同じピクセルである必要があります。ただし、常に表示されている必要はありません。一時的に消える、明るいコンテンツに隠れる、フェードイン・フェードアウトするウォーターマークも、設定可能な割合のフレーム数まで（`--frame_tolerance`、デフォルトは10%）検出・マスクされます。単一のマスクで表現できないのは、フレーム内を**移動する**ウォーターマークです。そのようなクリップはデータセットから除外してください。

マスクは完全に任意です。マスクファイルがないクリップは、これまでと同様にフレーム全体で学習されます。
</details>

## Mask files / マスクファイル

A mask is a grayscale image the same size as the source video (any resolution works — it is resized and cropped with the video, using the same transform, so it stays aligned with the latents).

| Pixel value | Meaning |
| --- | --- |
| `255` (white) | train on this pixel |
| `0` (black) | ignore this pixel (watermark) |

The mask lives next to the video, named after it with the `watermark_mask_suffix` appended (default `_wmask.png`):

```
dataset/
  clip_001.mp4
  clip_001.txt         # caption
  clip_001_wmask.png   # watermark mask
  clip_002.mp4
  clip_002.txt         # no mask: the whole frame is trained on
```

Image datasets use the same convention. If the mask is not found next to the source file, the cache directory is searched too.

<details>
<summary>日本語</summary>

マスクは、元の動画と同じサイズのグレースケール画像です（解像度は任意です。動画と同じ変換でリサイズ・クロップされるため、latentとの位置は一致します）。

| ピクセル値 | 意味 |
| --- | --- |
| `255`（白） | このピクセルを学習する |
| `0`（黒） | このピクセルを無視する（ウォーターマーク） |

マスクは動画と同じディレクトリに、動画のファイル名に `watermark_mask_suffix`（デフォルトは `_wmask.png`）を付けた名前で配置します（上記の例を参照）。

画像データセットでも同じ規則が使えます。ソースファイルの隣にマスクが見つからない場合は、キャッシュディレクトリも検索されます。
</details>

## Detecting the mask automatically / マスクの自動検出

`detect_watermark_mask.py` finds static watermarks by comparing sampled frames: a watermark pixel does not change over time, while the content underneath does. It samples frames evenly across each video, computes the per-pixel temporal standard deviation, and marks the low-variance pixels as watermark.

Before measuring that deviation it discards, per pixel, the frames furthest from that pixel's temporal median — up to `--frame_tolerance` of them. That is what makes an intermittent watermark detectable: without it, a handful of frames where the watermark is off screen would dominate the deviation and hide it completely.

### The second signal: semi-transparent watermarks

Deviation on its own only finds near-opaque overlays. A semi-transparent watermark blends as `alpha * W + (1 - alpha) * content`, so its pixels keep varying with the content underneath and their deviation is merely scaled by `1 - alpha`. At 30% opacity the deviation is still 70% of the content's — far above any threshold that would not also swallow every calm region of the frame. Measured on synthetic footage with a content deviation of 65:

| opacity | deviation at the watermark | found by deviation alone |
| --- | --- | --- |
| 0.3 | 45.6 | no |
| 0.5 | 33.8 | no |
| 0.7 | 21.8 | no |
| 0.95 | 6.8 | yes |

So a second pass looks at the **image gradient** instead. Content gradients change from frame to frame and cancel out in a temporal median, while the watermark contributes the same `alpha * grad(W)` in every one of them and survives. Whatever is left standing after that median is, by construction, what every frame has in common. (The same observation underpins Dekel et al., *On the Effectiveness of Visible Watermarks*, CVPR 2017.)

`--gradient_threshold` sets how far above the frame's own median that surviving gradient must be. Ordinary content and static background sit around 1x; a watermark between 20% and 50% opacity lands between 3x and 14x, so the default of 3.0 has margin on both sides. Raising `--n_frames` widens that gap, because content gradients cancel more completely the more frames are averaged.

The two signals are not ordered and there is no mode to pick: they are unioned on every clip. Neither covers the whole range alone — an opaque logo trips both, a 30% opacity logo only the gradient, a large flat opaque patch trips deviation strongly and the gradient only along its border.

The temporal median tolerates a watermark missing from a minority of frames on its own, so `--frame_tolerance` does not need to be raised for the gradient signal.

On footage with strongly repetitive motion the gradient signal has a false-positive floor — a few small boxes over content, around 1% of the frame in testing. `--corner_only` and `--min_area` are the mitigations, and over-masking a little content is far cheaper than leaving watermark in the loss.

### Boxes, not outlines

What the two signals mark — an outline from the gradient, solid pixels from the deviation — is not the region that has to be excluded. Each detected region is replaced by its **bounding box**.

That is deliberate. The VAE encoder mixes a neighbourhood of pixels into every latent cell, so a watermark bleeds into latents that its own pixels do not cover; a mask traced tightly around thin glyph strokes under-masks once it is downsampled to latent resolution. A box covers that neighbourhood, needs no morphological reconstruction of a logo's interior, and errs in the safe direction. It costs very little: on a corner logo, the box masks 2.5% of the frame where the exact glyph shapes would have masked 2.3%.

Boxes are per connected region, so two logos in opposite corners stay two boxes rather than one box spanning the frame.

### Letterboxing and burnt-in borders

A letterbox bar is static, carries no content, and would otherwise teach the model to generate black bars — so it is detected and masked, and that is the correct outcome, not a false positive. The same goes for a burnt-in border or frame.

Cropping such footage is still better than masking it, because the bars keep taking up frame area that the VAE encodes either way. And there is a limit: bars eat up to a third of the frame on their own, which is why `--max_coverage` defaults to 0.4 rather than something tighter. Beyond that the clip is left alone, on the assumption that the detection has stopped meaning anything (see below).

```bash
python src/musubi_tuner/detect_watermark_mask.py --video_dir /path/to/video_dir --corner_only
```

This writes `{video_stem}_wmask.png` next to each video. Only `opencv-python` and `numpy` are needed, so it can be run separately from training.

Review the generated masks before training — the detector is a heuristic, and a wrong mask silently removes real content from the loss.

### Command line options

```
--video_dir VIDEO_DIR
    Directory containing the videos (required)

--recursive
    Search --video_dir recursively

--n_frames N_FRAMES
    Number of frames sampled per video (default: 30)
    More frames make the detection more reliable on slow-moving footage, at the cost of speed,
    and let content gradients cancel out better, which is what makes a semi-transparent watermark
    stand out

--threshold THRESHOLD
    Temporal standard deviation below which a pixel is considered static (0-255 scale, default: 8.0)
    Raise it if parts of the watermark are missed, lower it if too much content is masked

--frame_tolerance FRAME_TOLERANCE
    Fraction of sampled frames allowed to not show the watermark (default: 0.1)
    Per pixel, that many of the most deviating frames are discarded before measuring the deviation
    Set to 0 to require the watermark in every sampled frame
    The fraction is over the *sampled* frames, not the source frames: if the watermark is off screen
    for 10% of a clip, an unlucky sampling stride can still land on many more of those frames than
    that, so raise --n_frames along with it if a known watermark is missed
    High values (above ~0.3) start letting slow-moving content pass as static, so raise it gradually

--gradient_threshold GRADIENT_THRESHOLD
    How far above the frame's own median a pixel's frame-to-frame-consistent gradient must be to
    count as a semi-transparent watermark edge (default: 3.0, i.e. 3x that median)
    Ordinary content and static background sit around 1x; a watermark at 20-50% opacity lands
    between 3x and 14x depending on opacity and --n_frames
    Lower it if a faint watermark is missed, raise it if static textured background is picked up

--corner_only
    Only look for the watermark in a band along the frame border
    Recommended: watermarks usually sit in a corner, and this avoids masking static background in the middle of the frame

--corner_margin CORNER_MARGIN
    Width of the border band for --corner_only, as a fraction of the frame size (default: 0.25)

--dilate DILATE
    Grow each detected box by N pixels (default: 3)
    Anti-aliased watermark edges blend into the content and still carry watermark signal

--min_area MIN_AREA
    Drop detected regions smaller than N pixels before boxing them (default: 32)

--max_coverage MAX_COVERAGE
    Skip the video if the detected region covers more than this fraction of the frame (default: 0.4)
    This catches locked-off shots, where "temporally static" no longer means "watermark"; those measure
    around 0.8. Letterbox bars are a legitimate detection and reach about a third of the frame on their
    own, so the default sits between the two

--suffix SUFFIX
    Mask file suffix (default: _wmask.png). Must match watermark_mask_suffix in the dataset config

--output_dir OUTPUT_DIR
    Write masks here instead of next to the videos

--overwrite
    Regenerate masks that already exist
```

<details>
<summary>日本語</summary>

`detect_watermark_mask.py` は、サンプリングしたフレームを比較して静的なウォーターマークを検出します。ウォーターマークのピクセルは時間的に変化しませんが、その下のコンテンツは変化します。スクリプトは各動画からフレームを均等にサンプリングし、ピクセルごとの時間方向の標準偏差を計算して、分散の低いピクセルをウォーターマークとしてマークします。

標準偏差を計算する前に、ピクセルごとに、そのピクセルの時間方向の中央値から最も離れたフレームを `--frame_tolerance` の割合まで除外します。これにより、断続的に表示されるウォーターマークも検出できます。この処理がないと、ウォーターマークが画面外にある数フレームが標準偏差を支配し、検出が完全に失敗してしまいます。

**半透明のウォーターマーク**：標準偏差だけでは、ほぼ不透明なオーバーレイしか検出できません。半透明のウォーターマークは `alpha * W + (1 - alpha) * コンテンツ` として合成されるため、下のコンテンツに応じて変化し続け、その偏差は `1 - alpha` 倍になるだけです。不透明度30%であれば、偏差はコンテンツの70%のままであり、これを拾えるしきい値では、動きの少ない領域もすべて拾ってしまいます。

そこで2つ目のパスでは、**画像の勾配**を見ます。コンテンツの勾配はフレームごとに変化するため時間方向の中央値で相殺されますが、ウォーターマークは毎フレーム同じ `alpha * grad(W)` を寄与するため残ります（Dekel et al., *On the Effectiveness of Visible Watermarks*, CVPR 2017 と同じ着想です）。

`--gradient_threshold` は、残った勾配がフレーム自身の中央値の何倍以上であればウォーターマークの輪郭とみなすかを指定します。通常のコンテンツや静止した背景は約1倍、不透明度20〜50%のウォーターマークは3〜14倍になるため、デフォルトの3.0は両側に余裕があります。`--n_frames` を増やすとコンテンツの勾配がより相殺され、この差が広がります。

2つの信号に順序はなく、モードの選択もありません。常に両方が適用され、和集合が取られます。どちらか一方では全範囲をカバーできないためです（不透明なロゴは両方に、不透明度30%のロゴは勾配のみに反応します）。時間方向の中央値は少数のフレームでウォーターマークが欠けていても影響を受けないため、勾配の信号のために `--frame_tolerance` を上げる必要はありません。

**輪郭ではなくボックス**：2つの信号が示すのは、勾配であれば輪郭、標準偏差であればその領域のピクセルであり、いずれも損失から除外すべき領域そのものではありません。そこで、検出された各領域はその**バウンディングボックス**に置き換えられます。VAEのエンコーダは周辺のピクセルを各latentセルに混ぜ込むため、ウォーターマークは自身のピクセルが覆っていないlatentにも影響します。細い文字の形に沿った厳密なマスクは、latent解像度に縮小した時点で不足します。ボックスであればその周辺までカバーでき、ロゴ内部の復元処理も不要で、安全側に倒れます。コストもわずかです（隅のロゴの場合、厳密な形状で2.3%、ボックスで2.5%）。ボックスは連結領域ごとに作られるため、対角の隅にある2つのロゴが1つの巨大なボックスに統合されることはありません。

**レターボックスと焼き込まれた枠**：レターボックスの帯は静的で、コンテンツを含まず、そのまま学習すればモデルは黒帯を生成するようになります。したがって検出・マスクされるのが正しい動作であり、誤検出ではありません。焼き込まれた枠も同様です。ただし、そのような素材はマスクするよりクロップするほうが望ましく（VAEは黒帯も含めてエンコードするため）、帯だけでフレームの3分の1に達することもあります。`--max_coverage` のデフォルトが0.4であるのはこのためです。

```bash
python src/musubi_tuner/detect_watermark_mask.py --video_dir /path/to/video_dir --corner_only
```

各動画の隣に `{動画名}_wmask.png` が書き出されます。必要なのは `opencv-python` と `numpy` だけなので、学習とは別に実行できます。

検出はヒューリスティックであり、誤ったマスクは実際のコンテンツを損失から取り除いてしまいます。学習前に生成されたマスクを確認してください。

コマンドラインオプションについては、上記の英語の説明を参照してください。特に `--corner_only` の使用を推奨します。ウォーターマークは通常隅にあるため、フレーム中央の静止した背景をマスクしてしまうことを防げます。
</details>

## Dataset configuration / データセットの設定

The suffix is configurable per dataset (or in `[general]`) with `watermark_mask_suffix`:

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
watermark_mask_suffix = "_wmask.png" # optional, this is the default

[[datasets]]
video_directory = "/path/to/video_dir"
cache_directory = "/path/to/cache_directory"
target_frames = [1, 25, 79]
frame_extraction = "head"
```

No training argument is needed: masks are picked up automatically when the files are present. Latent caching is unaffected, so masks can be added or regenerated without re-caching the dataset.

<details>
<summary>日本語</summary>

サフィックスは `watermark_mask_suffix` で、データセットごと（または `[general]`）に設定できます（上記のTOMLの例を参照）。

学習時の引数は不要です。マスクファイルが存在すれば自動的に使用されます。latentのキャッシュには影響しないため、キャッシュを再作成せずにマスクを追加・再生成できます。
</details>

## How the mask is applied / マスクの適用方法

The mask reaches the training batch as `watermark_mask`, a `(B, H, W)` tensor in [0, 1] at the bucket resolution. In the loss (`reduce_loss` in `src/musubi_tuner/training/trainer_base.py`) it is area-downsampled to the latent resolution and broadcast over channels and frames, so masking works with the VAE's temporal and spatial compression without any per-architecture handling:

```
loss = mse(pred, target)            # (B, C, T, H, W), reduction="none"
loss = (loss * mask).sum() / mask.expand_as(loss).sum()
```

Dividing by the mask weight keeps the loss scale comparable to the unmasked mean, so learning rates do not need to be retuned. A fully open mask reproduces `loss.mean()` exactly.

Because area downsampling is used, a latent cell that only partially overlaps the watermark is weighted by the fraction of it that is clean, rather than being dropped entirely.

Losses without a spatial layout (patchified image models, where the loss is a token sequence) cannot be masked; a warning is logged once and the mask is ignored.

<details>
<summary>日本語</summary>

マスクは `watermark_mask`（バケット解像度の `(B, H, W)`、値は [0, 1]）として学習バッチに渡されます。損失の計算時（`src/musubi_tuner/training/trainer_base.py` の `reduce_loss`）に、latentの解像度へarea補間で縮小され、チャンネルとフレーム方向にブロードキャストされます。そのため、VAEの時間方向・空間方向の圧縮に関係なく、アーキテクチャごとの処理なしで動作します。

マスクの重みで割ることで、損失のスケールはマスクなしの平均と同程度に保たれるため、学習率を調整し直す必要はありません。マスクが全面的に開いている場合は、`loss.mean()` と完全に一致します。

area補間を使用しているため、ウォーターマークと部分的にのみ重なるlatentのセルは、そのうちクリーンな部分の割合で重み付けされます（完全に除外されるわけではありません）。

空間的な配置を持たない損失（トークン列となるpatchify系の画像モデル）にはマスクを適用できません。この場合、警告が一度ログに出力され、マスクは無視されます。
</details>
