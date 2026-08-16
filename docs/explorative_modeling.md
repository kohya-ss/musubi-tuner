# Explorative Modeling And Forward XM

## English

### Sources

This integration addresses [musubi-tuner issue
#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019). Its behavioral
references are the [Forward Explorative Modeling paper
v1](https://arxiv.org/abs/2607.27372v1), the [project
page](https://explorative-modeling.github.io/), and the Apache-2.0 [official
implementation at pinned commit
`9d06ced61e2d2775a34782eb5830584ae4ef6094`](https://github.com/alexiglad/XM/commit/9d06ced61e2d2775a34782eb5830584ae4ef6094).

### Enabling The Modes

Standard `NetworkTrainer` LoRA entry points use:

```toml
xm_best_of_k = 2
```

or:

```powershell
accelerate launch src/musubi_tuner/wan_train_network.py --config_file training.toml --xm_best_of_k 2
```

MiniMax-H3 uses one count and a multi-frame stream selector:

```toml
h3_best_of_k = 2
h3_best_of_k_stream = "video" # or "audio" for multi-frame batches
```

```powershell
accelerate launch src/musubi_tuner/minimax_h3_train_network.py --config_file h3_training.toml --h3_best_of_k 2 --h3_best_of_k_stream video
```

The former experimental `--h3_video_best_of_k` spelling was removed before
merge. Replace it with `--h3_best_of_k K --h3_best_of_k_stream video`.
The similarly named audio/image count options are not aliases and are rejected
with migration guidance.

Both count options accept integers of at least 1. For both, `K = 1` disables
exploration and uses the ordinary training dispatch. It performs no extra
candidate draw, forward, metric, or RNG operation.

### Standard Forward XM Contract

For `xm_best_of_k > 1`, the trainer evaluates K noise candidates sequentially
without gradients and recomputes the per-sample winners once with gradients.
Clean latents/data, timestep, conditioning, condition-drop decisions, and
other stochastic forward choices stay fixed. Each prediction target remains
paired with the noise that produced that candidate. Winners are selected
independently per sample with the architecture's actual weighted per-sample
training loss.

The sequential implementation retains one candidate and one `[B, ...]` winner
tensor rather than K activation graphs. It performs K no-grad forwards plus
one ordinary gradient-enabled forward. Relative to an approximate
forward-plus-backward cost of three forward operations, its operation-count
multiplier is `(K + 3) / 3`. Wall-clock cost depends on the model, hardware,
offloading, checkpointing, and compiler settings.

#### Legacy Weighted 4D Loss Correction

Independently of exploration, the approved legacy weighted rank-4 image-loss
fix changes an invalid cross-broadcast of a `[B, 1, 1, 1, 1]` weight against a
`[B, C, H, W]` loss into one weight per image sample. Consequently, affected
weighted 4D `K = 1` values can differ from upstream dev. This narrow correction
does not change unweighted 4D losses or weighted 5D losses.

### MiniMax-H3 Stream-Focused Best-Of-K

For a multi-frame batch, `h3_best_of_k_stream` selects the noise and raw MSE
used for ranking: `video` varies video noise and ranks video MSE, while `audio`
varies audio noise and ranks audio MSE. The other stream, base time, both
shifted times, conditions, and stochastic forward choices stay fixed. A
one-frame image batch always varies video noise and reports the runtime kind as
`image`, regardless of the configured multi-frame stream.

The winner is always recomputed with the ordinary joint
`video_loss + effective_audio_weight * audio_loss` objective. Image selection,
and video selection when the effective audio weight is zero, therefore rank
the complete effective objective. Otherwise video- or audio-focused selection
is a heuristic: the selected-stream winner need not minimize the joint loss.

An audio-stream batch with zero effective audio weight cannot be ranked. It
falls back to one ordinary prepared forward, with no candidate seed draw or
exploration metrics. `--video_only` with active audio search is rejected at
startup; `--audio_loss_weight 0` is allowed and logs the fallback warning.
MiniMax-H3 rejects `--xm_best_of_k > 1`.

### Compatibility

Standard Forward XM supports these LoRA entry points:

- `flux_2_train_network.py`
- `flux_kontext_train_network.py`
- `fpack_train_network.py`
- `hv_train_network.py`
- `hv_1_5_train_network.py`
- `ideogram4_train_network.py`
- `kandinsky5_train_network.py`
- `krea2_train_network.py`
- `qwen_image_train_network.py`
- `wan_train_network.py`
- `zimage_train_network.py`

`flux_2_train_network_self_flow.py` supports standard XM only when
`--self_flow` is off. Enabled Self-Flow is rejected because its
teacher/student candidate state and objective are not implemented. HiDream-O1
is rejected because its candidate-local noising transform cannot be rebuilt by
the standard affine path and its optional DINO term is batch-reduced.
MiniMax-H3 rejects standard Forward XM because its H3-specific heuristic is
implemented through an architecture-specific joint video/audio path.

Standalone full-finetune entry points such as `hv_train.py`,
`qwen_image_train.py`, `zimage_train.py`, and `hidream_o1_train.py` do not use
the shared `NetworkTrainer` loop and are unchanged.

### Metrics And Numerical Failure Policy

When enabled, standard XM logs `xm/candidate_loss_mean` and
`xm/selection_gain`. H3 logs exactly one runtime-kind pair:
`h3_best_of_k/{image|video|audio}/candidate_loss_mean` and
`h3_best_of_k/{image|video|audio}/selection_gain`. For `xm/`, the selection
score is the architecture's actual weighted per-sample training loss. For H3,
it is the raw per-sample MSE of the selected runtime stream.

For either prefix, `candidate_loss_mean` is the mean selection score over every
candidate and every sample. `selection_gain` is the mean candidate-zero
selection score minus the mean selected-winner selection score. It is not the
candidate mean minus a winner mean. Neither exploration metric is a cross-K
quality metric. `K = 1` follows ordinary dispatch and emits none of these
exploration metrics.

For an H3 final update, `loss/current` is the final composite scalar. The
winner's `loss/video` and `loss/audio` are unweighted component means;
`audio_loss_weight` is applied when composing `loss/current` (through the
effective per-sample audio weight, including audio-supervision gating). Thus the
H3 exploration metrics describe selected-stream candidate ranking, not
necessarily the final video-plus-weighted-audio objective.

Forward XM and H3 best-of-K raise before backward when any selected candidate
score is NaN or infinite. H3 does not add a separate fail-fast check to the
non-selected component during search. It does not silently discard a candidate
whose selected score is non-finite.
This is intentionally stricter than ordinary mixed-precision training, where
GradScaler or another loss-scaling configuration may detect non-finite
gradients during backward and skip an update. Do not assume the modes recover
from numerical instability in the same way.

### Comparing K Values

Training loss is not directly comparable across K because candidate selection
changes its distribution. Published Forward XM improvements are pretraining
results; this integration establishes no LoRA quality, convergence, or
data-efficiency gain. Guidance and inference are unchanged.

Start with a controlled `K = 1` versus `K = 2` experiment using the same seed,
data, optimizer-step budget, and downstream validation metric. Do not choose a
run by comparing raw training loss across K. There is no universal recommended
K.

### Validated Runtime

R1 is tested on Python 3.10.11, PyTorch `2.13.0+cu130`, CUDA 13.0, and an NVIDIA
GeForce RTX 4090 with compute capability 8.9. This is the current validation
runtime only; `cu124` is not in this feature's test matrix. The implementation
uses existing public PyTorch APIs and adds no version gate, but other runtimes
are outside this test matrix. Stable APIs may work elsewhere; that is not a
compatibility claim.

## 日本語

### 参照資料

この実装は [musubi-tuner issue
#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019) に対応します。動作上の
参照元は [Forward Explorative Modeling 論文
v1](https://arxiv.org/abs/2607.27372v1)、[プロジェクトページ](https://explorative-modeling.github.io/)、
および Apache-2.0 の [固定済み commit
`9d06ced61e2d2775a34782eb5830584ae4ef6094` にある公式実装](https://github.com/alexiglad/XM/commit/9d06ced61e2d2775a34782eb5830584ae4ef6094) です。

### 有効化

標準の `NetworkTrainer` LoRA エントリポイントでは次を指定します。

```toml
xm_best_of_k = 2
```

```powershell
accelerate launch src/musubi_tuner/wan_train_network.py --config_file training.toml --xm_best_of_k 2
```

MiniMax-H3 では候補数と multi-frame 用 stream を指定します。

```toml
h3_best_of_k = 2
h3_best_of_k_stream = "video" # multi-frame の音声探索では "audio"
```

```powershell
accelerate launch src/musubi_tuner/minimax_h3_train_network.py --config_file h3_training.toml --h3_best_of_k 2 --h3_best_of_k_stream video
```

以前の experimental な `--h3_video_best_of_k` は merge 前に削除されました。
`--h3_best_of_k K --h3_best_of_k_stream video` に置き換えてください。同様の名前の
audio/image 用 count オプションも alias ではなく、移行方法を示して拒否されます。

どちらの count も 1 以上の整数を受け付けます。どちらも `K = 1` では探索を
無効化し、通常の学習 dispatch をそのまま使います。追加の候補生成、forward、
メトリクス、RNG 操作は発生しません。

### 標準 Forward XM の契約

`xm_best_of_k > 1` では K 個のノイズ候補を勾配なしで逐次評価し、サンプルごとの
勝者だけを勾配ありで一度再計算します。clean latent/data、timestep、conditioning、
condition drop、その他の確率的 forward 条件は候補間で固定されます。prediction
target は必ず、その候補を生成したノイズと組にしたまま扱います。勝者はモデルの
実際の重み付き per-sample 学習 loss で、サンプルごとに独立して選択されます。

実装は K 個の activation graph を保持せず、現在の候補と `[B, ...]` の勝者だけを
保持します。計算は K 回の no-grad forward と 1 回の通常の勾配あり forward です。
forward + backward をおよそ 3 forward 相当とみなした演算回数倍率は
`(K + 3) / 3` です。実時間はモデル、GPU、offload、checkpoint、compile 設定に
依存します。

#### 従来の重み付き 4D loss の修正

探索機能とは独立して、承認済みの従来の重み付き rank-4 image loss 修正は、
`[B, C, H, W]` loss に対する `[B, 1, 1, 1, 1]` weight の不正な cross-broadcast を、
画像サンプルごとに一つの weight とする処理へ変更します。したがって影響を受ける
重み付き 4D の `K = 1` 値は upstream dev と異なることがあります。この限定的な
修正は、重みなし 4D loss と重み付き 5D loss を変更しません。

### MiniMax-H3 の stream 重視 best-of-K

multi-frame batch では `h3_best_of_k_stream` が探索するノイズと順位付け用 raw MSE を
選びます。`video` は動画ノイズと動画 MSE、`audio` は音声ノイズと音声 MSE を使います。
もう一方の stream、base time、両方の shifted time、condition、確率的 forward 条件は
固定されます。one-frame image batch は設定された multi-frame stream に関係なく必ず
動画ノイズを探索し、runtime kind は `image` になります。

winner は常に通常の joint objective
`video_loss + effective_audio_weight * audio_loss` で再計算されます。したがって image
探索、および effective audio weight が 0 の video 探索は完全な effective objective と
同じ順位になります。それ以外の video/audio stream 重視探索はヒューリスティックで、
選んだ stream の winner が joint loss を最小にするとは限りません。

effective audio weight が 0 の audio batch は順位付けできないため、候補 seed や探索
メトリクスを作らず、準備済みの通常 forward 1 回へ fallback します。active な audio
探索と `--video_only` の組み合わせは起動時に拒否されます。`--audio_loss_weight 0` は
許可され、fallback warning を出します。MiniMax-H3 は `--xm_best_of_k > 1` を拒否します。

### 対応範囲

標準 Forward XM は次の LoRA エントリポイントに対応します。

- `flux_2_train_network.py`
- `flux_kontext_train_network.py`
- `fpack_train_network.py`
- `hv_train_network.py`
- `hv_1_5_train_network.py`
- `ideogram4_train_network.py`
- `kandinsky5_train_network.py`
- `krea2_train_network.py`
- `qwen_image_train_network.py`
- `wan_train_network.py`
- `zimage_train_network.py`

`flux_2_train_network_self_flow.py` は `--self_flow` が off の通常 Flux 2 経路だけを
サポートします。Self-Flow 有効時は teacher/student の候補状態と目的関数が未実装の
ため拒否します。HiDream-O1 は候補ごとの noising 変換を標準 affine 経路から再構成
できず、任意の DINO 項も batch reduction なので拒否します。MiniMax-H3 は標準 XM を
拒否します。H3 の joint video/audio 探索は architecture 固有の経路で実装されます。

`hv_train.py`、`qwen_image_train.py`、`zimage_train.py`、
`hidream_o1_train.py` などの full-finetune エントリポイントは共有
`NetworkTrainer` loop を使わないため変更しません。

### メトリクスと非有限 loss の方針

有効時、標準 XM は `xm/candidate_loss_mean` と `xm/selection_gain` を記録します。
H3 は runtime kind に応じた一組だけを記録します。
`h3_best_of_k/{image|video|audio}/candidate_loss_mean` と
`h3_best_of_k/{image|video|audio}/selection_gain` です。`xm/` の選択 score は、各
アーキテクチャの実際の重み付き per-sample 学習 loss です。H3 の選択 score は、選択した
runtime stream の raw per-sample MSE です。

どちらの prefix でも `candidate_loss_mean` はすべての候補とすべての sample にわたる
selection score の平均です。`selection_gain` は candidate-zero の selection score の
平均から、選択された winner の selection score の平均を引いた値です。candidate mean
から winner mean を引いたものではありません。どちらの探索メトリクスも cross-K の
quality metric ではありません。`K = 1` は通常 dispatch を使うため、これらの探索
メトリクスは記録しません。

H3 の最終 update では、`loss/current` は最終 composite scalar です。winner の
`loss/video` と `loss/audio` は重みなしの component mean であり、
`audio_loss_weight` は `loss/current` を構成するときに適用されます（audio supervision
の gating を含む effective per-sample audio weight を通じて）。したがって H3 の探索
メトリクスは選択 stream の順位付けを示し、必ずしも最終の video + weighted audio
objective を示すものではありません。

Forward XM と H3 best-of-K は、選択対象の候補 score に NaN または infinity が一つでも
あれば backward 前に例外を送出します。H3 は探索中の非選択 component に別の fail-fast
検査を追加しません。選択 score が非有限の候補を黙って捨てることはしません。
これは通常の mixed-precision 学習より厳しい方針です。通常経路では GradScaler や
loss scaling 設定によって backward 中に非有限 gradient を検出し、更新を skip する
場合があります。両経路が数値不安定から同じように回復するとは考えないでください。

### K の比較

候補選択によって分布が変わるため、K が異なる run の training loss は直接比較
できません。論文の改善結果は pretraining の結果であり、この実装は LoRA の品質、
収束、data efficiency の改善を保証しません。guidance と inference は変わりません。

同一の seed、data、optimizer step 数、downstream validation metric を使い、まず
`K = 1` と `K = 2` を比較してください。異なる K の raw training loss で run を
選ばないでください。すべての環境に通用する推奨 K はありません。

### 検証環境

R1 は Python 3.10.11、PyTorch `2.13.0+cu130`、CUDA 13.0、compute capability 8.9 の
NVIDIA GeForce RTX 4090 で検証済みです。これは現在の検証 runtime のみであり、`cu124`
はこの機能の test matrix に含まれません。既存の public PyTorch API だけを使い
version gate は追加しませんが、その他の runtime はこの test matrix の対象外です。
stable API により別の環境で動作する可能性はありますが、互換性の主張ではありません。
