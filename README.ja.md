## Blissful Tuner

[English](./README.md) | [日本語](./README.ja.md)

Blyss Sarania による Musubi Tuner の Blissful な拡張機能

(このセクションは機械翻訳です)

ここでは、生成動画モデルを扱うためのツールスイートの作成に重点を置いた、高度で実験的な機能を備えたMusubi Tunerの拡張バージョンをご覧いただけます。動画生成時にプレビューしたり、推論速度を向上させたり、動画を長くしたり、作成した動画をより細かく制御したり、VFIやアップスケーリングなどで動画を強化したりできます。Musubiをさらに活用したい方は、ぜひこの機会にお試しください。最適なパフォーマンスと互換性を得るには、Python 3.12とPyTorch 2.7.0以降を推奨します。「requirements.txt」に追加の要件が追加されているため、通常のMusubiから移行する場合は、再度`pip install -r requirements.txt`を実行する必要があります。開発はPython 3.12で行われていますが、3.10との互換性も維持するよう努めています。

Musubi Tunerの開発に尽力いただいたkohya-ssさん、重要なコードを移植したHunyuanVideoWrapperとWanVideoWrapperを開発してくださったkijaiさん、そしてオープンソース生成AIコミュニティの開発者の皆様に心より感謝申し上げます。多くの変更は実験的なものであるため、修正前のMusubiと同じように動作しない部分もあることをご了承ください。何か問題が見つかった場合はお知らせください。できる限り修正いたします。このバージョンに関する問題は、MusubiのメインGithubリポジトリではなく、このリポジトリのIssuesセクションに投稿してください。
すべてのモデル/モード向けの拡張機能:
- 美しく豊富なログ、豊富な引数解析、豊富なトレースバック

すべてのモデル向けの拡張機能：
- latent2RGBまたはTAEHVによる生成中に潜在プレビュー（`--preview_latent_every N`、Nはステップ数（フレームパックの場合はセクション数）。デフォルトではlatent2rgbを使用しますが、TAEは`--preview_vae /path/to/model`で有効にできます。モデル：https://huggingface.co/Blyss/BlissfulModels/tree/main/taehv ）
- 高速で高品質な生成のための最適化された生成設定（`--optimized`、モデルに基づいてさまざまな最適化と設定を有効にします。SageAttention、Triton、PyTorch 2.7.0以降が必要です）
- 動画/画像に生成メタデータを保存します (`--container mkv` で自動的に保存され、PNG を保存する場合は `--no-metadata` で無効になり、`--container mp4` では使用できません。`blissful_tuner/metaview.py some_video.mkv` を使用すると、このようなメタデータを簡単に表示/コピーできます)
- 拡張された保存オプション (`--codec codec --container container`、Apple ProRes (`--codec prores`、超高ビットレートの知覚的ロスレス) を `--container mkv` に保存、または `h264`、`h265` のいずれかを `mp4` または `mkv` に保存可能)
- FP16 積算 (`--fp16_accumulation`、Wan FP16 モデルで最も効果的に機能します (Hunyaun bf16 でも機能します!)。PyTorch 2.7.0 以上が必要ですが、推論速度が大幅に向上します。特に `--compile` を使用すると、fp8_fast/mmscaled とほぼ同等の速度になります。精度の低下は抑えられています！fp8スケールモードにも対応しています！
- シードとして文字列を使用するのは良いでしょう！覚えやすいのも魅力です！
- プロンプトでワイルドカードを使用すると、バリエーションが増えます！（`--prompt_wildcards /path/to/wildcard/directory` のように指定します。例えば、プロンプトで `__color__` と指定すると、そのディレクトリ内の color.txt が検索されます。ワイルドカードファイルの形式は、1行につき1つの置換文字列で、red:2.0 や "some long string:0.5" のように相対的な重みを任意で付加できます。ワイルドカード自体にワイルドカードを含めることも可能で、再帰回数の制限は50回です！）

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
- 正規化アテンションガイダンス (NAG) (https://arxiv.org/pdf/2505.21179) (クロスアテンションレイヤー内でネガティブガイダンスを提供します。通常の CFG だけでなく、蒸留モデルでも機能します。`--nag_scale 3.0` で有効にして、ネガティブプロンプトを提供してください!)
- 高品質かつ低ステップの蒸留サンプリング (https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill のような蒸留 Wan モデル/LoRA で `--sample_solver lcm` または `--sample_solver dpm++sde` を使用します。また、便利な LoRA も作成しました: https://huggingface.co/Blyss/BlissfulModels/tree/main/wan_lcm )
- V2V 推論 (`--video_path /path/to/input/video --denoise_strength amount`、amount は 0.0 - 1.0 の浮動小数点数で、ソース ビデオに追加されるノイズの強さを制御します。`--noise_mode traditional` の場合、他の実装と同様に、タイム ステップ スケジュールの最後の (amount * 100) パーセントが実行されます。`--noise_mode direct` の場合、タイム ステップ スケジュール内でその値に最も近い場所から開始して、追加されるノイズの量を可能な限り直接制御します。スケーリング、パディング、切り捨てをサポートしているため、入力は出力と同じ解像度や長さである必要はありません。`--video_length` が入力より短い場合、入力は切り捨てられ、最初の `--video_length` フレームのみが含まれます。`--video_length` が入力より長い場合、最初のフレームまたは最後のフレームが繰り返され、長さが埋められます。 `--v2v_pad_mode` に依存します。T2V または I2V の `--task` モードとモデルを使用できます（個人的には i2v モードの方が品質が良いと思います）。I2V モードでは、`--image_path` が指定されていない場合、代わりにビデオの最初のフレームがモデルの調整に使用されます。`--infer_steps` は、完全なノイズ除去と同じ値にする必要があります。例えば、T2V の場合はデフォルトで 50、I2V の場合は 40 です。これは、完全なスケジュールから変更する必要があるためです。実際のステップ数は `--noise_mode` に依存します。
- I2I推論 (`--i2i_path /path/to/image` - T2IモードでT2Vモデルと共に使用し、`--denoise_strength`で強度を指定します。潜在ノイズの増強のための`--i2_extra_noise`もサポートします)
- プロンプトの重み付け（`--prompt_weighting` を指定し、プロンプトで「(large:1.4) の赤いボールで遊ぶ猫」のように記述することで、「large」の効果を強調できます。[this] または (this) に注意してください。はサポートされておらず、(this:1.0) のみサポートされています。また、重み付けのダウンウェイトには奇妙な効果があります。
- 複素数を使用しない ComfyUI から移植された ROPE。`--compile` と併用すると VRAM を大幅に節約できます！(`--rope_func comfy`)
- I2V/V2V 用のオプションの潜在ノイズ (`--v2v_extra_noise 0.02 --i2v_extra_noise 0.02`、0.04 未満の値を推奨。これにより V2V/I2V のディテールとテクスチャが向上しますが、値が大きすぎるとアーティファクトや影の動きが発生します。V2V では 0.01～0.02、I2V では 0.02～0.04 程度を使用しています)
- 混合精度トランスフォーマーをロードします (推論またはトレーニングの場合は `--mixed_precision_transformer` を使用します。このようなトランスフォーマーの作成方法とその理由については、https://github.com/kohya-ss/musubi-tuner/discussions/232#discussioncomment-13284677 を参照してください)。

フレームパックのみの拡張機能:
- Torch.compile (`--compile`、Wan と Hunyuan が既に使用している構文と同じ)
- FP8 fast/mm_scaled (`--fp8_fast` は、40xx カードで若干の品質低下を伴いますが、速度が向上します。Wan と Hunyuan は既にネイティブ Musubi でこの機能を搭載しています！)

機種に依存しない追加機能:
(以下のスクリプトを使用する場合は、`--group postprocess` (例: すべての要件を完全にインストールするには、`pip install -e . --group postprocess --group dev`) を使用してプロジェクトを venv にインストールしてください。)
- GIMM-VFI フレームレート補間 (`blissful_tuner/GIMMVFI.py`。使用方法については `--help` を参照してください。対応モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/VFI )
- SwinIR または ESRGAN タイプのモデルによるアップスケーリング (`blissful_tuner/upscaler.py`。使用方法については `--help` を参照してください。対応モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/upscaling )
- Yoloベースの顔ぼかしスクリプト - 顔の改変を伴わないLoRAのトレーニングに役立ちます！(`blissful_tuner/yolo_blur.py`、使用方法については`--help`をご覧ください。推奨モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/yolo )
- CodeFormer/GFPGANによる顔の修復 (`blissful_tuner/facefix.py`、いつものように`--help` を見てください! モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/face_restoration )

また、私の関連プロジェクト（ https://github.com/Sarania/Envious ）は、LinuxのターミナルからNvidia GPUを管理するのに便利です。nvidia-ml-pyが必要ですが、リアルタイムモニタリング、オーバークロック/アンダークロック、電力制限調整、ファン制御、プロファイルなどをサポートしています。GPU VRAM用の小さなプロセスモニターも付いています！nvidia-smiがダメな場合のnvidia-smiのようなものです😂

私のコード全体と Musubi Tuner のコードは Apache 2.0 ライセンスです。含まれている他のプロジェクトはライセンスが異なる場合があります。その場合は、それぞれのディレクトリにライセンス条項を記載した LICENSE ファイルがあります。以下は、現在でも有効なオリジナルの Musubi Readme です。

# Musubi Tuner

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー](#スポンサー)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [最近の更新](#最近の更新)
    - [リリースについて](#リリースについて)
    - [AIコーディングエージェントを使用する開発者の方へ](#AIコーディングエージェントを使用する開発者の方へ)
- [概要](#概要)
    - [ハードウェア要件](#ハードウェア要件)
    - [特徴](#特徴)
- [インストール](#インストール)
    - [pipによるインストール](#pipによるインストール)
    - [uvによるインストール](#uvによるインストール)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
- [モデルのダウンロード](#モデルのダウンロード)
- [使い方](#使い方)
    - [データセット設定](#データセット設定)
    - [事前キャッシュと学習](#事前キャッシュと学習)
    - [Accelerateの設定](#Accelerateの設定)
    - [学習と推論](#学習と推論)
- [その他](#その他)
    - [SageAttentionのインストール方法](#SageAttentionのインストール方法)
    - [PyTorchのバージョンについて](#PyTorchのバージョンについて)
- [免責事項](#免責事項)
- [コントリビューションについて](#コントリビューションについて)
- [ライセンス](#ライセンス)
</details>

## はじめに

このリポジトリは、HunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-ImageのLoRA学習用のコマンドラインツールです。このリポジトリは非公式であり、公式のHunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-Imageのリポジトリとは関係ありません。

アーキテクチャ固有のドキュメントについては、以下を参照してください：
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [FramePack](./docs/framepack.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

*リポジトリは開発中です。*

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 最近の更新

GitHub Discussionsを有効にしました。コミュニティのQ&A、知識共有、技術情報の交換などにご利用ください。バグ報告や機能リクエストにはIssuesを、質問や経験の共有にはDiscussionsをご利用ください。[Discussionはこちら](https://github.com/kohya-ss/musubi-tuner/discussions)

- 2025/09/06
    - 新しいLRスケジューラRexを追加しました。[PR #513](https://github.com/kohya-ss/musubi-tuner/pull/513) xzuyn氏に感謝します。
        - powerを1未満に設定した Polynomial Scheduler に似ていますが、Rexは学習率の減少がより緩やかです。
        - 詳細は[高度な設定のドキュメント](./docs/advanced_config.md#rex)を参照してください。

- 2025/09/02 (update)
    - Qwen-Imageのfine tuningに対応しました。[PR #492](https://github.com/kohya-ss/musubi-tuner/pull/492)
        - LoRA学習ではなくモデル全体を学習します。詳細は[Qwen-Imageのドキュメントのfinetuningの節](./docs/qwen_image.md#finetuning)を参照してください。

- 2025/09/02
    - ruffによるコード解析を導入しました。[PR #483](https://github.com/kohya-ss/musubi-tuner/pull/483) および[PR #488](https://github.com/kohya-ss/musubi-tuner/pull/488) arledesma 氏に感謝します。
        - ruffはPythonのコード解析、整形ツールです。
    - コードの貢献をいただく際は、`ruff check`を実行してコードスタイルを確認していただけると助かります。`ruff --fix`で自動修正も可能です。
        - なおコードの整形はblackで行うか、ruffのblack互換フォーマットを使い、`line-length`を`132`に設定してください。
        - ガイドライン等をのちほど整備する予定です。
    
- 2025/08/28
    - RTX 50シリーズのGPUをお使いの場合、PyTorch 2.8.0をお試しください。
    - ライブラリの依存関係を更新し、`bitsandbytes`からバージョン指定を外しました。環境に応じた適切なバージョンをインストールしてください。
        - RTX 50シリーズのGPUを使用している場合は、`pip install -U bitsandbytes`で最新バージョンをインストールするとエラーが解消されます。
        - `sentencepiece`を0.2.1に更新しました。
    - [Schedule Free Optimizer](https://github.com/facebookresearch/schedule_free)をサポートしました。PR [#505](https://github.com/kohya-ss/musubi-tuner/pull/505) am7coffee氏に感謝します。
        - [Schedule Free Optimizerのドキュメント](./docs/advanced_config.md#schedule-free-optimizer--スケジュールフリーオプティマイザ)を参照してください。

- 2025/08/24
    - Wan2.1/2.2の学習、推論時のピークメモリ使用量を削減しました。PR [#493](https://github.com/kohya-ss/musubi-tuner/pull/493) 動画のフレームサイズ、フレーム数にもよりますが重み以外のメモリ使用量が10%程度削減される可能性があります。

- 2025/08/22
    - Qwen-Image-Editに対応しました。PR [#473](https://github.com/kohya-ss/musubi-tuner/pull/473) 詳細は[Qwen-Imageのドキュメント](./docs/qwen_image.md)を参照してください。変更が多岐に渡るため既存機能へ影響がある可能性があります。不具合が発生した場合は、[Issues](https://github.com/kohya-ss/musubi-tuner/issues)でご報告ください。
    - **破壊的変更**: この変更に伴いFLUX.1 Kontextのキャッシュフォーマットが変更されました。Latentキャッシュを再作成してください。

- 2025/08/18
    - `qwen_image_train_network.py`の訓練時の`--network_module networks.lora_qwen_image`の指定について、ドキュメントへの記載が漏れていました。[ドキュメント](./docs/qwen_image.md#training--学習)を修正しました。

- 2025/08/16
    - Qwen-ImageのVLMを利用したキャプション生成ツールを追加しました。PR [#460](https://github.com/kohya-ss/musubi-tuner/pull/460) 詳細は[ドキュメント](./docs/tools.md#image-captioning-with-qwen25-vl-srcmusubi_tunercaption_images_by_qwen_vlpy)を参照してください。

- 2025/08/15
    - Timestep Bucketing機能が追加されました。これにより、タイムステップの分布がより均一になり、学習が安定します。PR [#418](https://github.com/kohya-ss/musubi-tuner/pull/418) 詳細は[Timestep Bucketingのドキュメント](./docs/advanced_config.md#timestep-bucketing-for-uniform-sampling--均一なサンプリングのためのtimestep-bucketing)を参照してください。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

### AIコーディングエージェントを使用する開発者の方へ

このリポジトリでは、ClaudeやGeminiのようなAIエージェントが、プロジェクトの概要や構造を理解しやすくするためのエージェント向け文書（プロンプト）を用意しています。

これらを使用するためには、プロジェクトのルートディレクトリに各エージェント向けの設定ファイルを作成し、明示的に読み込む必要があります。

**セットアップ手順:**

1.  プロジェクトのルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリが推奨するプロンプトをインポートします（現在、両者はほぼ同じ内容です）：

    ```markdown
    @./.ai/claude.prompt.md
    ```

    Geminiの場合はこちらです：

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  インポートした行の後に、必要な指示を適宜追加してください（例：`Always respond in Japanese.`）。

このアプローチにより、共有されたプロジェクトのコンテキストを活用しつつ、エージェントに与える指示を各ユーザーが自由に制御できます。`CLAUDE.md` と `GEMINI.md` はすでに `.gitignore` に記載されているため、リポジトリにコミットされることはありません。

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
pip install -e .
```

オプションとして、FlashAttention、SageAttention（**推論にのみ使用できます**、インストール方法は[こちら](#SageAttentionのインストール方法)を参照）を使用できます。

また、`ascii-magic`（データセットの確認に使用）、`matplotlib`（timestepsの可視化に使用）、`tensorboard`（学習ログの記録に使用）、`prompt-toolkit`を必要に応じてインストールしてください。

`prompt-toolkit`をインストールするとWan2.1およびFramePackのinteractive modeでの編集に、自動的に使用されます。特にLinux環境でプロンプトの編集が容易になります。

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
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

モデルのダウンロード手順はアーキテクチャによって異なります。各アーキテクチャの詳細については、以下のドキュメントを参照してください：

- [HunyuanVideoのモデルダウンロード](./docs/hunyuan_video.md#download-the-model--モデルのダウンロード)
- [Wan2.1/2.2のモデルダウンロード](./docs/wan.md#download-the-model--モデルのダウンロード)
- [FramePackのモデルダウンロード](./docs/framepack.md#download-the-model--モデルのダウンロード)
- [FLUX.1 Kontextのモデルダウンロード](./docs/flux_kontext.md#download-the-model--モデルのダウンロード)
- [Qwen-Imageのモデルダウンロード](./docs/qwen_image.md#download-the-model--モデルのダウンロード)

## 使い方

### データセット設定

[こちら](./src/musubi_tuner/dataset/dataset_config.md)を参照してください。

### 事前キャッシュと学習

各アーキテクチャは固有の事前キャッシュと学習手順が必要です。詳細については、以下のドキュメントを参照してください：

- [HunyuanVideoの使用方法](./docs/hunyuan_video.md)
- [Wan2.1/2.2の使用方法](./docs/wan.md)
- [FramePackの使用方法](./docs/framepack.md)
- [FLUX.1 Kontextの使用方法](./docs/flux_kontext.md)
- [Qwen-Imageの使用方法](./docs/qwen_image.md)

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

### 学習と推論

学習と推論の手順はアーキテクチャによって大きく異なります。詳細な手順については、対応するドキュメントを参照してください：

- [HunyuanVideoの学習と推論](./docs/hunyuan_video.md)
- [Wan2.1/2.2の学習と推論](./docs/wan.md)
- [FramePackの学習と推論](./docs/framepack.md)
- [FLUX.1 Kontextの学習と推論](./docs/flux_kontext.md)
- [Qwen-Imageの学習と推論](./docs/qwen_image.md)

高度な設定オプションや追加機能については、以下を参照してください：
- [高度な設定](./docs/advanced_config.md)
- [学習中のサンプル生成](./docs/sampling_during_training.md)
- [ツールとユーティリティ](./docs/tools.md)

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

このリポジトリは非公式であり、サポートされているアーキテクチャの公式リポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

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
