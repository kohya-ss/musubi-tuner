## Blissful Tuner

[English](./README.md) | [日本語](./README.ja.md)

Blyss Sarania による Musubi Tuner の Blissful な拡張機能

(このセクションは機械翻訳です)

ここでは、生成動画モデルを扱うためのツールスイートの作成に重点を置いた、高度で実験的な機能を備えたMusubi Tunerの拡張バージョンをご覧いただけます。動画生成時にプレビューしたり、推論速度を向上させたり、動画を長くしたり、作成した動画をより細かく制御したり、VFIやアップスケーリングなどで動画を強化したりできます。Musubiをさらに活用したい方は、ぜひこの機会にお試しください。最適なパフォーマンスと互換性を得るには、Python 3.12とPyTorch 2.7.0以降を推奨します。「requirements.txt」に追加の要件が追加されているため、通常のMusubiから移行する場合は、再度`pip install -r requirements.txt`を実行する必要があります。開発はPython 3.12で行われていますが、3.10との互換性も維持するよう努めています。

Musubi Tunerの開発に尽力いただいたkohya-ssさん、重要なコードを移植したHunyuanVideoWrapperとWanVideoWrapperを開発してくださったkijaiさん、そしてオープンソース生成AIコミュニティの開発者の皆様に心より感謝申し上げます。多くの変更は実験的なものであるため、修正前のMusubiと同じように動作しない部分もあることをご了承ください。何か問題が見つかった場合はお知らせください。できる限り修正いたします。このバージョンに関する問題は、MusubiのメインGithubリポジトリではなく、このリポジトリのIssuesセクションに投稿してください。

すべてのモデル向けの拡張機能：
- latent2RGBまたはTAEHVによる生成中に潜在プレビュー（`--preview_latent_every N`、Nはステップ数（フレームパックの場合はセクション数）。デフォルトではlatent2rgbを使用しますが、TAEは`--preview_vae /path/to/model`で有効にできます。モデル：https://www.dropbox.com/scl/fi/fxkluga9uxu5x6xa94vky/taehv.7z?rlkey=ux1vmcg1yk78gv7iy4iqznpn7&st=4181tzkp&dl=0）
- 高速で高品質な生成のための最適化された生成設定（`--optimized`、モデルに基づいてさまざまな最適化と設定を有効にします。SageAttention、Triton、PyTorch 2.7.0以降が必要です）
- 動画/画像に生成メタデータを保存します (`--container mkv` で自動、`--no-metadata` で無効化、`--container mp4` では使用できません。`blissful_tuner/metaview.py some_video.mkv` でこのようなメタデータを表示/コピーできます)
- 美しくリッチなログ出力、豊富な引数解析、そして豊富なトレースバック
- 拡張された保存オプション (`--codec codec --container container`、Apple ProRes (`--codec prores`、超高ビットレートの知覚的ロスレス) を `--container mkv` に保存、または `h264`、`h265` のいずれかを `mp4` または `mkv` に保存可能)
- FP16 積算 (`--fp16_accumulation`、Wan FP16 モデルで最も効果的に機能します (Hunyaun bf16 でも機能します!)。PyTorch 2.7.0 以上が必要ですが、推論速度が大幅に向上します。特に `--compile` を使用すると、fp8_fast/mmscaled とほぼ同等の速度になります。精度の低下は抑えられています！fp8スケールモードにも対応しています！
- シードとして文字列を使用するのは良いでしょう！覚えやすいのも魅力です！
- プロンプトでワイルドカードを使用すると、バリエーションが増えます！（`--prompt_wildcards /path/to/wildcard/directory` のように指定します。例えば、プロンプトで `__color__` と指定すると、そのディレクトリ内の color.txt が検索されます。ワイルドカードファイルの形式は、1行につき1つの置換文字列で、red:2.0 や "some long string:0.5" のように相対的な重みを任意で付加できます。ワイルドカード自体にワイルドカードを含めることも可能で、再帰回数の制限は50回です！）
- 精度向上のため、量子化/線形変換をアップキャストします (`--upcast_quantization` は推論とトレーニングの両方ですべてのモデルで使用可能で、わずかな VRAM コストで fp8_scaled 量子化の精度をわずかに向上します。`--upcast_linear` は推論中(トレーニングは近日中に)にすべてのモデルで使用可能で、fp8_scaled を使用すると、乗算ヘッドルームのために線形変換を fp32 にアップキャストします。ペナルティはほとんど発生せず、品質がわずかに向上します。scaled_mm (fp8_fast) が有効になっているレイヤーには適用されません)

Wan/Hunyuan 拡張機能：
- 拡散パイプ形式の LoRA を、事前に変換することなく推論用に読み込みます。
- RifleX など。より長い動画は https://github.com/thu-ml/RIFLEx をご覧ください（`--riflex_index N` で、N は RifleX の周波数です。Wan の場合は 6 が適しており、通常 81 フレームではなく約 115 フレームまで再生できます。Wan の場合は `--rope_func comfy` が必要です。Hunyuan の場合は 4 が適しており、少なくとも 2 倍の長さにできます！）
- CFGZero* 例: https://github.com/WeichenFan/CFG-Zero-star (`--cfgzerostar_scaling --cfgzerostar_init_steps N` で、N は開始時に 0 になるまでのステップ数です。T2V の場合は 2、I2V の場合は 1 が適切ですが、私の経験では T2V の方が適しています。Hunyuan のサポートは非​​常に実験的であり、CFG が有効になっている場合にのみ利用可能です。)
- 高度な CFG スケジューリング: (`--cfg_schedule`、使用方法については `--help` を参照してください。必要に応じて、個々のステップにガイダンスのスケールダウンを指定することもできます!)
- 垂直負ガイダンス (`--perp_neg neg_strength`、neg_strength は負プロンプトの文字列を制御する浮動小数点数です。詳細は `--help` を参照してください!)

Hunyuan 専用の拡張機能:
- その他の LLM オプション(`--hidden_​​state_skip_layer N --apply_final_norm --reproduce`、説明は`--help`を参照してください！)
- Wanと同じアルゴリズムを使用したFP8スケールのサポート（`--fp8_scaled`、推論と学習の両方に強く推奨。より優れたfp8なので、これだけ知っておく必要があります！）
- CLIP用の別のプロンプト（`--prompt_2 "second prompt goes here"`、CLIPはよりシンプルなテキストに使用されるため、異なるプロンプトを提供します）
- https://github.com/zer0int/ComfyUI-HunyuanVideo-Nyanに基づいてテキストエンコーダーを再スケール（`--te_multiplier llm clip`、例えば`--te_multiplier 0.9 1.2`のように、LLMをわずかに重み付け下げ、CLIPをわずかに重み付け上げます）

WAN 専用拡張機能（ワンショットモードとインタラクティブモードの両方をサポート）：
- V2V 推論（`--video_path /path/to/input/video --v2v_denoise amount` で、amount は 0.0 - 1.0 の浮動小数点数で、ソースビデオに追加するノイズの強度を制御します。`--v2v_noise_mode Traditional` の場合、他の実装と同様に、タイムステップスケジュールの最後の (amount * 100) パーセントを実行します。`--v2v_noise_mode direct` の場合、タイムステップスケジュール内でその値に最も近い場所から開始し、そこから処理を進めることで、追加するノイズの量を可能な限り正確に直接制御します。スケーリング、パディング、切り捨てをサポートしているため、入力は出力と同じ解像度や長さである必要はありません。`--video_length` が入力より短い場合、入力は切り捨てられ、最初の `--video_length` フレームのみが含まれます。 `--video_length` が入力より長い場合、`--v2v_pad_mode` の設定に応じて、最初のフレームまたは最後のフレームが繰り返され、長さが調整されます。T2V または I2V の `--task` モードとモデルを使用できます（個人的には i2v モードの方が品質が良いと思います）。I2V モードでは、`--image_path` が指定されていない場合、代わりにビデオの最初のフレームがモデルの調整に使用されます。`--infer_steps` は、完全なノイズ除去の場合と同じ値にする必要があります。例えば、T2V の場合はデフォルトで 50、I2V の場合は 40 です。これは、完全なスケジュールから変更する必要があるためです。実際のステップ数は `--v2v_noise_mode` の設定に依存します。
- プロンプトの重み付け（`--prompt_weighting` を指定し、プロンプトで「(large:1.4) の赤いボールで遊ぶ猫」のように記述することで、「large」の効果を強調できます。[this] または (this) に注意してください。はサポートされておらず、(this:1.0) のみサポートされています。また、重み付けのダウンウェイトには奇妙な効果があります。
- 複素数を使用しない ComfyUI から移植された ROPE。`--compile` と併用すると VRAM を大幅に節約できます！(`--rope_func comfy`)
- I2V/V2V 用のオプションの潜在ノイズ (`--v2v_extra_noise 0.02 --i2v_extra_noise 0.02`、0.04 未満の値を推奨。これにより V2V/I2V のディテールとテクスチャが向上しますが、値が大きすぎるとアーティファクトや影の動きが発生します。V2V では 0.01～0.02、I2V では 0.02～0.04 程度を使用しています)
- 混合精度トランスフォーマーをロードします (推論またはトレーニングの場合は `--mixed_precision_transformer` を使用します。このようなトランスフォーマーの作成方法とその理由については、https://github.com/kohya-ss/musubi-tuner/discussions/232#discussioncomment-13284677 を参照してください)。

フレームパックのみの拡張機能:
- Torch.compile (`--compile`、Wan と Hunyuan が既に使用している構文と同じ)
- FP8 fast/mm_scaled (`--fp8_fast` は、40xx カードで若干の品質低下を伴いますが、速度が向上します。Wan と Hunyuan は既にネイティブ Musubi でこの機能を搭載しています！)

機種に依存しない追加機能:
- GIMM-VFI フレームレート補間 (`blissful_tuner/GIMMVFI.py`。使用方法については `--help` を参照してください。対応モデル: https://www.dropbox.com/scl/fi/tcq68jxr52o2gi47eup37/gimm-vfi.7z?rlkey=skvzwxi9lv9455py5wrxv6r5j&st=gu5einkd&dl=0 )
- SwinIR または ESRGAN タイプのモデルによるアップスケーリング (`blissful_tuner/upscaler.py`。使用方法については `--help` を参照してください。対応モデル: https://www.dropbox.com/scl/fi/wh5hw55o8rofg5mal9uek/upscale.7z?rlkey=oom3osa1zo0pf55092xcfnjp1&st=dozwpzwk&dl=0 )
- Yoloベースの顔ぼかしスクリプト - 顔の改変を伴わないLoRAのトレーニングに役立ちます！(`blissful_tuner/yolo_blur.py`、使用方法については`--help`をご覧ください。推奨モデル: https://www.dropbox.com/scl/fi/44xdsohltv2kofxrirvrj/yolo.7z?rlkey=zk6bv5iw3ic1pbgo4e8cblbw1&st=kwm6fzgk&dl=0 )
- CodeFormer/GFPGANによる顔の修復 (`blissful_tuner/facefix.py`、いつものように`--help` を見てください! モデル: https://www.dropbox.com/scl/fi/0ylqy170w0lpwwvb4acvx/facefix.7z?rlkey=25bljmfw95p9pn899upres0d7&st=ho29pd6d&dl=0 )

また、私の関連プロジェクト（ https://github.com/Sarania/Envious ）は、LinuxのターミナルからNvidia GPUを管理するのに便利です。nvidia-ml-pyが必要ですが、リアルタイムモニタリング、オーバークロック/アンダークロック、電力制限調整、ファン制御、プロファイルなどをサポートしています。GPU VRAM用の小さなプロセスモニターも付いています！nvidia-smiがダメな場合のnvidia-smiのようなものです😂

私のコード全体と Musubi Tuner のコードは Apache 2.0 ライセンスです。含まれている他のプロジェクトはライセンスが異なる場合があります。その場合は、それぞれのディレクトリにライセンス条項を記載した LICENSE ファイルがあります。以下は、現在でも有効なオリジナルの Musubi Readme です。

# Musubi Tuner

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [最近の更新](#最近の更新)
    - [リリースについて](#リリースについて)
- [概要](#概要)
    - [ハードウェア要件](#ハードウェア要件)
    - [特徴](#特徴)
- [インストール](#インストール)
- [モデルのダウンロード](#モデルのダウンロード)
    - [HunyuanVideoの公式モデルを使う](#HunyuanVideoの公式モデルを使う)
    - [Text EncoderにComfyUI提供のモデルを使う](#Text-EncoderにComfyUI提供のモデルを使う)
- [使い方](#使い方)
    - [データセット設定](#データセット設定)
    - [latentの事前キャッシュ](#latentの事前キャッシュ)
    - [Text Encoder出力の事前キャッシュ](#Text-Encoder出力の事前キャッシュ)
    - [Accelerateの設定](#Accelerateの設定)
    - [学習](#学習)
    - [LoRAの重みのマージ](#LoRAの重みのマージ)
    - [推論](#推論)
    - [SkyReels V1での推論](#SkyReels-V1での推論)
    - [LoRAの形式の変換](#LoRAの形式の変換)
- [その他](#その他)
    - [SageAttentionのインストール方法](#SageAttentionのインストール方法)
- [免責事項](#免責事項)
- [コントリビューションについて](#コントリビューションについて)
- [ライセンス](#ライセンス)
</details>

## はじめに

このリポジトリは、HunyuanVideo、Wan2.1、FramePackのLoRA学習用のコマンドラインツールです。このリポジトリは非公式であり、公式のHunyuanVideoやWan2.1、FramePackのリポジトリとは関係ありません。

Wan2.1については、[Wan2.1のドキュメント](./docs/wan.md)も参照してください。FramePackについては、[FramePackのドキュメント](./docs/framepack.md)を参照してください。

*リポジトリは開発中です。*

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。


### 最近の更新

- GitHub Discussionsを有効にしました。コミュニティのQ&A、知識共有、技術情報の交換などにご利用ください。バグ報告や機能リクエストにはIssuesを、質問や経験の共有にはDiscussionsをご利用ください。[Discussionはこちら](https://github.com/kohya-ss/musubi-tuner/discussions)

- 2025/05/30
    - データセットの読み込み時にリサイズが正しく行われない場合がある不具合を修正しました。キャッシュの再作成をお願いします。PR [#312](https://github.com/kohya-ss/musubi-tuner/issues/312) sdbds 氏に感謝します。
        - リサイズ前の画像の幅または高さがバケットの幅または高さと一致していて、かつもう片方が異なる場合（具体的には、たとえば元画像が640\*480で、バケットが640\*360の場合など）に不具合が発生していました。
    - FramePackの1フレーム推論、学習のコードを大幅に改良しました。詳細は[FramePackの1フレーム推論のドキュメント](./docs/framepack_1f.md)を参照してください。
        - **破壊的変更**: 1フレーム学習のデータセット形式、学習オプション、推論オプションが変更されました。ドキュメントに従って、データセット設定の変更、キャッシュの再作成、学習・推論オプションの変更を行ってください。
    - FramePackの1フレーム推論と学習についてのドキュメントを追加しました。詳細は[前述のドキュメント](./docs/framepack_1f.md)を参照してください。

- 2025/05/22
    - FramePackの推論スクリプトで、以下の対応を行いました。
        - **破壊的変更**: 1フレーム推論で画像を保存する場合、サブディレクトリを作成しなくなりました。
        - バッチモードとインタラクティブモードに対応しました。
            - バッチモードでは、プロンプトをファイルから読み込んで生成します。インタラクティブモードでは、コマンドラインからプロンプトを指定して生成します。詳細は[FramePackのドキュメント](./docs/framepack.md#batch-and-interactive-modes--バッチモードとインタラクティブモード)を参照してください。
        - kisekaeichi方式の参照画像を複数指定できるようになりました。詳細は[FramePackのドキュメント](./docs/framepack.md#kisekaeichi-method-history-reference-options--kisekaeichi方式履歴参照オプション)を参照してください。

- 2025/05/17 update 1
    - `--max_data_loader_n_workers`に2以上を指定するとひとつのエポック内でデータの重複や欠落が起きる不具合を修正しました。PR [#287](https://github.com/kohya-ss/musubi-tuner/pull/287),  issue [#283](https://github.com/kohya-ss/musubi-tuner/issues/283)
        - 長期的には全てのデータが学習されますが、短期的にはデータの偏りが起きていました。
        - データセットの初期化が不適切でそれぞれのDataSetが異なる順番でデータを返していたため、複数のDataLoaderの使用時に不具合が起きていました。初期化を修正してすべてのDataSetが同じ順番でデータを返すよう修正しました。

- 2025/05/17
    - FramePackの1フレーム推論でkisekaeichi方式に対応しました。furusu氏の提案したこの新しい推論方式は、post latentに参照画像を設定することで生成される画像を制御するものです。詳細は[FramePackのドキュメント](./docs/framepack.md#kisekaeichi-method-history-reference-options--kisekaeichi方式履歴参照オプション)を参照してください。

- 2025/05/11
    - FramePackの学習で1フレーム推論用の学習に対応しました（実験的機能）。詳細は[FramePackのドキュメント](./docs/framepack.md#single-frame-training--1フレーム学習)を参照してください。
    
- 2025/05/09 update 2
    - FramePackの推論コードで、1フレーム推論に対応しました。これは当リポジトリの独自の機能で、動画ではなく、プロンプトに従って時間経過した後の画像を生成するものです。つまり、限定的ですが画像の自然言語による編集が可能です。詳細は[FramePackのドキュメント](./docs/framepack.md#single-frame-inference--単一フレーム推論)を参照してください。
    - FramePackの推論コードに、生成する動画長を秒数ではなくセクション数で指定する`--video_sections`オプションを追加しました。また`--output_type latent_images`（latentと画像の両方を保存）が追加されました。

- 2025/05/09 
    - FramePackの推論コードで、HunyuanVideo用のLoRAを適用できるようになりました。当リポジトリのLoRAとdiffusion-pipeのLoRAの両方が適用可能です。詳細は[FramePackのドキュメント](./docs/framepack.md#inference)を参照してください。

- 2025/05/04
    - FramePack-F1の学習および推論を追加しました（実験的機能）。詳細は[FramePackのドキュメント](./docs/framepack.md)を参照してください。
        - FramePack-F1用に、`--f1`オプションを指定してlatentのキャッシュを再作成してください（`--vanilla_sampling`が`--f1`に変わり、仕様も変わっています）。FramePack-F1はFramePackとは互換性がありません。FramePackとFramePack-F1のキャッシュファイルは共有できないため、別の`.toml`ファイルを使用して別のキャッシュディレクトリを指定してください。

- 2025/05/01
    - FramePackの推論コードに、latent padding指定、カスタムプロンプト指定等の機能を追加しました。詳細は[FramePackのドキュメント](./docs/framepack.md#inference)を参照してください。
        - セクション開始画像を指定したときの振る舞いが変わりました（latent paddingを自動的に0に指定しなくなったため、開始画像は参照画像として用いられます）。以前と同じ振る舞い（セクション開始画像を強制）にするには、`--latent_padding 0,0,0,0`（セクション数だけ0を指定）としてください。
- 2025/04/26
    - FramePackの推論およびLoRA学習を追加しました。PR [#230](https://github.com/kohya-ss/musubi-tuner/pull/230) 詳細は[FramePackのドキュメント](./docs/framepack.md)を参照してください。
    
- 2025/04/18
    - Wan2.1の推論時に、ファイルからプロンプトを読み込んで生成する一括生成モードと、コマンドラインからプロンプトを指定して生成するインタラクティブモードを追加しました。詳細は[こちら](./docs/wan.md#interactive-mode--インタラクティブモード)を参照してください。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

## 概要

### ハードウェア要件

- VRAM: 静止画での学習は12GB以上推奨、動画での学習は24GB以上推奨。
    - *解像度等の学習設定により異なります。*12GBでは解像度 960x544 以下とし、`--blocks_to_swap`、`--fp8_llm`等の省メモリオプションを使用してください。
- メインメモリ: 64GB以上を推奨、32GB+スワップで動作するかもしれませんが、未検証です。

### 特徴

- 省メモリに特化
- Windows対応（Linuxでの動作報告もあります）
- マルチGPUには対応していません

## インストール

### pipによるインストール

Python 3.10以上を使用してください（3.10で動作確認済み）。

適当な仮想環境を作成し、ご利用のCUDAバージョンに合わせたPyTorchとtorchvisionをインストールしてください。

PyTorchはバージョン2.5.1以上を使用してください（[補足](#PyTorchのバージョンについて)）。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

以下のコマンドを使用して、必要な依存関係をインストールします。

```bash
pip install -r requirements.txt
```

オプションとして、FlashAttention、SageAttention（**推論にのみ使用できます**、インストール方法は[こちら](#SageAttentionのインストール方法)を参照）を使用できます。

また、`ascii-magic`（データセットの確認に使用）、`matplotlib`（timestepsの可視化に使用）、`tensorboard`（学習ログの記録に使用）を必要に応じてインストールしてください。

```bash
pip install ascii-magic matplotlib tensorboard
```
### uvによるインストール

uvを使用してインストールすることもできますが、uvによるインストールは試験的なものです。フィードバックを歓迎します。

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

表示される指示に従い、pathを設定してください。

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

表示される指示に従い、PATHを設定するか、この時点でシステムを再起動してください。

## モデルのダウンロード

以下のいずれかの方法で、モデルをダウンロードしてください。

### HunyuanVideoの公式モデルを使う 

[公式のREADME](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md)を参考にダウンロードし、任意のディレクトリに以下のように配置します。

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Text EncoderにComfyUI提供のモデルを使う

こちらの方法の方がより簡単です。DiTとVAEのモデルはHumyuanVideoのものを使用します。

https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers から、[mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のモデルもありますが、未検証です。）

`--fp8_base`を指定して学習する場合は、`mp_rank_00_model_states.pt`の代わりに、[こちら](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial)の`mp_rank_00_model_states_fp8.safetensors`を使用可能です。（このファイルは非公式のもので、重みを単純にfloat8_e4m3fnに変換したものです。）

また、https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae から、[pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) をダウンロードし、任意のディレクトリに配置します。

Text EncoderにはComfyUI提供のモデルを使用させていただきます。[ComyUIのページ](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)を参考に、https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders から、llava_llama3_fp16.safetensors （Text Encoder 1、LLM）と、clip_l.safetensors （Text Encoder 2、CLIP）をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のLLMモデルもありますが、動作未検証です。）

## 使い方

### データセット設定

[こちら](./dataset/dataset_config.md)を参照してください。

### latentの事前キャッシュ

latentの事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。（pipによるインストールの場合）

```bash
python cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

uvでインストールした場合は、`uv run python cache_latents.py ...`のように、`uv run`を先頭につけてください。以下のコマンドも同様です。

その他のオプションは`python cache_latents.py --help`で確認できます。

VRAMが足りない場合は、`--vae_spatial_tile_sample_min_size`を128程度に減らし、`--batch_size`を小さくしてください。

`--debug_mode image` を指定するとデータセットの画像とキャプションが新規ウィンドウに表示されます。`--debug_mode console`でコンソールに表示されます（`ascii-magic`が必要）。

`--debug_mode video`で、キャッシュディレクトリに画像または動画が保存されます（確認後、削除してください）。動画のビットレートは確認用に低くしてあります。実際には元動画の画像が学習に使用されます。

`--debug_mode`指定時は、実際のキャッシュ処理は行われません。

デフォルトではデータセットに含まれないキャッシュファイルは自動的に削除されます。`--keep_cache`を指定すると、キャッシュファイルを残すことができます。

### Text Encoder出力の事前キャッシュ

Text Encoder出力の事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。

```bash
python cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

その他のオプションは`python cache_text_encoder_outputs.py --help`で確認できます。

`--batch_size`はVRAMに合わせて調整してください。

VRAMが足りない場合（16GB程度未満の場合）は、`--fp8_llm`を指定して、fp8でLLMを実行してください。

デフォルトではデータセットに含まれないキャッシュファイルは自動的に削除されます。`--keep_cache`を指定すると、キャッシュファイルを残すことができます。

### Accelerateの設定

`accelerate config`を実行して、Accelerateの設定を行います。それぞれの質問に、環境に応じた適切な値を選択してください（値を直接入力するか、矢印キーとエンターで選択、大文字がデフォルトなので、デフォルト値でよい場合は何も入力せずエンター）。GPU 1台での学習の場合、以下のように答えてください。

```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`、つまり1台目のGPUが使われます。）

### 学習

以下のコマンドを使用して、学習を開始します（実際には一行で入力してください）。

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

__更新__：サンプルの学習率を1e-3から2e-4に、`--timestep_sampling`を`sigmoid`から`shift`に、`--discrete_flow_shift`を1.0から7.0に変更しました。より高速な学習が期待されます。ディテールが甘くなる場合は、discrete flow shiftを3.0程度に下げてみてください。

ただ、適切な学習率、学習ステップ数、timestepsの分布、loss weightingなどのパラメータは、以前として不明な点が数多くあります。情報提供をお待ちしています。

その他のオプションは`python hv_train_network.py --help`で確認できます（ただし多くのオプションは動作未確認です）。

`--fp8_base`を指定すると、DiTがfp8で学習されます。未指定時はmixed precisionのデータ型が使用されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。`--fp8_base`を指定しない場合はVRAM 24GB以上を推奨します。また必要に応じて`--blocks_to_swap`を使用してください。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大36が指定できます。

（block swapのアイデアは2kpr氏の実装に基づくものです。2kpr氏にあらためて感謝します。）

`--sdpa`でPyTorchのscaled dot product attentionを使用します。`--flash_attn`で[FlashAttention]:(https://github.com/Dao-AILab/flash-attention)を使用します。`--xformers`でxformersの利用も可能ですが、xformersを使う場合は`--split_attn`を指定してください。`--sage_attn`でSageAttentionを使用しますが、SageAttentionは現時点では学習に未対応のため、エラーが発生します。

`--split_attn`を指定すると、attentionを分割して処理します。速度が多少低下しますが、VRAM使用量はわずかに減ります。

学習されるLoRAの形式は、`sd-scripts`と同じです。

`--show_timesteps`に`image`（`matplotlib`が必要）または`console`を指定すると、学習時のtimestepsの分布とtimestepsごとのloss weightingが確認できます。

学習時のログの記録が可能です。[TensorBoard形式のログの保存と参照](./docs/advanced_config.md#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照)を参照してください。

PyTorch Dynamoによる最適化を行う場合は、[こちら](./docs/advanced_config.md#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化)を参照してください。

`--gradient_checkpointing`を指定すると、gradient checkpointingが有効になります。VRAM使用量は減りますが、学習速度は低下します。

`--optimizer_type`には`adamw8bit`、`adamw8bit_apex_fused`、`adamw8bit_apex_fused_legacy`、`adamw8bit_apex_fused_legacy_no_scale`のいずれかを指定してください。

学習中のサンプル画像生成については、[こちらのドキュメント](./docs/sampling_during_training.md)を参照してください。その他の高度な設定については[こちらのドキュメント](./docs/advanced_config.md)を参照してください。

### LoRAの重みのマージ

注：Wan 2.1には対応していません。

```bash
python merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

`--device`には計算を行うデバイス（`cpu`または`cuda`等）を指定してください。`cuda`を指定すると計算が高速化されます。

`--lora_weight`にはマージするLoRAの重みを、`--lora_multiplier`にはLoRAの重みの係数を、それぞれ指定してください。複数個が指定可能で、両者の数は一致させてください。

### 推論

以下のコマンドを使用して動画を生成します。

```bash
python hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

その他のオプションは`python hv_generate_video.py --help`で確認できます。

`--fp8`を指定すると、DiTがfp8で推論されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。

RTX 40x0シリーズのGPUを使用している場合は、`--fp8_fast`オプションを指定することで、高速推論が可能です。このオプションを指定する場合は、`--fp8`も指定してください。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大38が指定できます。

`--attn_mode`には`flash`、`torch`、`sageattn`、`xformers`または`sdpa`（`torch`指定時と同じ）のいずれかを指定してください。それぞれFlashAttention、scaled dot product attention、SageAttention、xformersに対応します。デフォルトは`torch`です。SageAttentionはVRAMの削減に有効です。

`--split_attn`を指定すると、attentionを分割して処理します。SageAttention利用時で10%程度の高速化が見込まれます。

`--output_type`には`both`、`latent`、`video`、`images`のいずれかを指定してください。`both`はlatentと動画の両方を出力します。VAEでOut of Memoryエラーが発生する場合に備えて、`both`を指定することをお勧めします。`--latent_path`に保存されたlatentを指定し、`--output_type video` （または`images`）としてスクリプトを実行すると、VAEのdecodeのみを行えます。

`--seed`は省略可能です。指定しない場合はランダムなシードが使用されます。

`--video_length`は「4の倍数+1」を指定してください。

`--flow_shift`にタイムステップのシフト値（discrete flow shift）を指定可能です。省略時のデフォルト値は7.0で、これは推論ステップ数が50の時の推奨値です。HunyuanVideoの論文では、ステップ数50の場合は7.0、ステップ数20未満（10など）で17.0が推奨されています。

`--video_path`に読み込む動画を指定すると、video2videoの推論が可能です。動画ファイルを指定するか、複数の画像ファイルが入ったディレクトリを指定してください（画像ファイルはファイル名でソートされ、各フレームとして用いられます）。`--video_length`よりも短い動画を指定するとエラーになります。`--strength`で強度を指定できます。0~1.0で指定でき、大きいほど元の動画からの変化が大きくなります。

なおvideo2video推論の処理は実験的なものです。

`--compile`オプションでPyTorchのコンパイル機能を有効にします（実験的機能）。tritonのインストールが必要です。また、WindowsではVisual C++ build toolsが必要で、かつPyTorch>=2.6.0でのみ動作します。`--compile_args`でコンパイル時の引数を渡すことができます。

`--compile`は初回実行時にかなりの時間がかかりますが、2回目以降は高速化されます。

`--save_merged_model`オプションで、LoRAマージ後のDiTモデルを保存できます。`--save_merged_model path/to/merged_model.safetensors`のように指定してください。なおこのオプションを指定すると推論は行われません。

### SkyReels V1での推論

SkyReels V1のT2VとI2Vモデルがサポートされています（推論のみ）。

モデルは[こちら](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy)からダウンロードできます。モデルを提供してくださったKijai氏に感謝します。`skyreels_hunyuan_i2v_bf16.safetensors`がI2Vモデル、`skyreels_hunyuan_t2v_bf16.safetensors`がT2Vモデルです。`bf16`以外の形式は未検証です（`fp8_e4m3fn`は動作するかもしれません）。

T2V推論を行う場合、以下のオプションを推論コマンドに追加してください：

```bash
--guidance_scale 6.0 --embedded_cfg_scale 1.0 --negative_prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" --split_uncond
```

SkyReels V1はclassifier free guidance（ネガティブプロンプト）を必要とするようです。`--guidance_scale`はネガティブプロンプトのガイダンススケールです。公式リポジトリの推奨値は6.0です。デフォルトは1.0で、この場合はclassifier free guidanceは使用されません（ネガティブプロンプトは無視されます）。

`--embedded_cfg_scale`は埋め込みガイダンスのスケールです。公式リポジトリの推奨値は1.0です（埋め込みガイダンスなしを意味すると思われます）。

`--negative_prompt`はいわゆるネガティブプロンプトです。上記のサンプルは公式リポジトリのものです。`--guidance_scale`を指定し、`--negative_prompt`を指定しなかった場合は、空文字列が使用されます。

`--split_uncond`を指定すると、モデル呼び出しをuncondとcond（ネガティブプロンプトとプロンプト）に分割します。VRAM使用量が減りますが、推論速度は低下する可能性があります。`--split_attn`が指定されている場合、`--split_uncond`は自動的に有効になります。

### LoRAの形式の変換

ComfyUIで使用可能な形式（Diffusion-pipeと思われる）への変換は以下のコマンドで行えます。

```bash
python convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

`--input`と`--output`はそれぞれ入力と出力のファイルパスを指定してください。

`--target`には`other`を指定してください。`default`を指定すると、他の形式から当リポジトリの形式に変換できます。

Wan2.1も対応済みです。

## その他

### SageAttentionのインストール方法

sdbds氏によるWindows対応のSageAttentionのwheelが https://github.com/sdbds/SageAttention-for-windows で公開されています。triton をインストールし、Python、PyTorch、CUDAのバージョンが一致する場合は、[Releases](https://github.com/sdbds/SageAttention-for-windows/releases)からビルド済みwheelをダウンロードしてインストールすることが可能です。sdbds氏に感謝します。

参考までに、以下は、SageAttentionをビルドしインストールするための簡単な手順です。Microsoft Visual C++ 再頒布可能パッケージを最新にする必要があるかもしれません。

1. Pythonのバージョンに応じたtriton 3.1.0のwhellを[こちら](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5)からダウンロードしてインストールします。

2. Microsoft Visual Studio 2022かBuild Tools for Visual Studio 2022を、C++のビルドができるよう設定し、インストールします。（上のRedditの投稿を参照してください）。

3. 任意のフォルダにSageAttentionのリポジトリをクローンします。
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. スタートメニューから Visual Studio 2022 内の `x64 Native Tools Command Prompt for VS 2022` を選択してコマンドプロンプトを開きます。

5. venvを有効にし、SageAttentionのフォルダに移動して以下のコマンドを実行します。DISTUTILSが設定されていない、のようなエラーが出た場合は `set DISTUTILS_USE_SDK=1`としてから再度実行してください。
    ```shell
    python setup.py install
    ```

以上でSageAttentionのインストールが完了です。

### PyTorchのバージョンについて

`--attn_mode`に`torch`を指定する場合、2.5.1以降のPyTorchを使用してください（それより前のバージョンでは生成される動画が真っ黒になるようです）。

古いバージョンを使う場合、xformersやSageAttentionを使用してください。

## 免責事項

このリポジトリは非公式であり、公式のHunyuanVideoリポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

- 実際の稼働環境での動作を意図したものではありません
- 機能やAPIは予告なく変更されることがあります
- いくつもの機能が未検証です
- 動画学習機能はまだ開発中です

問題やバグについては、以下の情報とともにIssueを作成してください：

- 問題の詳細な説明
- 再現手順
- 環境の詳細（OS、GPU、VRAM、Pythonバージョンなど）
- 関連するエラーメッセージやログ

## コントリビューションについて

コントリビューションを歓迎します。ただし、以下にご注意ください：

- メンテナーのリソースが限られているため、PRのレビューやマージには時間がかかる場合があります
- 大きな変更に取り組む前には、議論のためのIssueを作成してください
- PRに関して：
    - 変更は焦点を絞り、適度なサイズにしてください
    - 明確な説明をお願いします
    - 既存のコードスタイルに従ってください
    - ドキュメントが更新されていることを確認してください

## ライセンス

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

`wan`ディレクトリ以下のコードは、[Wan2.1](https://github.com/Wan-Video/Wan2.1)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

`frame_pack`ディレクトリ以下のコードは、[frame_pack](https://github.com/lllyasviel/FramePack)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
