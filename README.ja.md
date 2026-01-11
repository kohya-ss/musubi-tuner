## Blissful Tuner

[English](./README.md) | [日本語](./README.ja.md)

Blyss Sarania による Musubi Tuner の Blissful な拡張機能

(このセクションは機械翻訳です)

ここでは、生成動画モデルを扱うためのツールスイートの作成に重点を置いた、高度で実験的な機能を備えたMusubi Tunerの拡張バージョンをご覧いただけます。動画生成時にプレビューしたり、推論速度を向上させたり、動画を長くしたり、作成した動画をより細かく制御したり、VFIやアップスケーリングなどで動画を強化したりできます。Musubiをさらに活用したい方は、ぜひこの機会にお試しください。最適なパフォーマンスと互換性を得るには、Python 3.12とPyTorch 2.7.0以降を推奨します。「requirements.txt」に追加の要件が追加されているため、通常のMusubiから移行する場合は、再度`pip install -r requirements.txt`を実行する必要があります。開発はPython 3.12で行われていますが、3.10との互換性も維持するよう努めています。

重要事項：MusubiとBlissfulを切り替える際は、同じvenvに通常のMusubi TunerまたはBlissful Tunerのいずれか一方のみをインストールし、既存のものをアンインストールしてください（例：`pip uninstall blissful-tuner`）。Blissful TunerはMusubi Tunerの上に直接構築されており、多くのファイルを共有しています。この手順を踏まずに切り替えると、多くの問題が発生する可能性があります。よろしくお願いいたします。

Musubi Tunerの開発に尽力いただいたkohya-ssさん、重要なコードを移植したHunyuanVideoWrapperとWanVideoWrapperを開発してくださったkijaiさん、そしてオープンソース生成AIコミュニティの開発者の皆様に心より感謝申し上げます。多くの変更は実験的なものであるため、修正前のMusubiと同じように動作しない部分もあることをご了承ください。何か問題が見つかった場合はお知らせください。できる限り修正いたします。このバージョンに関する問題は、MusubiのメインGithubリポジトリではなく、このリポジトリのIssuesセクションに投稿してください。

プロジェクトの拡大に​​合わせてこのセクションをメンテナンスしやすくするため、各機能は一度だけリストアップし、プロジェクト内のどのモデルが現在その機能をサポートしているかを示す凡例も表示します。ほとんどの機能は推論に関するもので、トレーニングに利用可能な機能がある場合は特に明記します。また、記載しきれないほど多くの小さな最適化や機能追加も実施しました。最新のアップデートについては、[こちら](https://github.com/kohya-ss/musubi-tuner/discussions/232) で開発ログのようなものを公開しています。

現在のモデルの凡例：Hunyuan Video: (HY)、Wan 2.1/2.2: (WV)、Framepack: (FP)、Flux (FX)、Qwen Image (QI)、トレーニングに利用可能: (T)

素晴らしい機能：
- 美しく豊富なログ機能、豊富なargparse、豊富なトレースバック (HY) (WV) (FP) (FX) (QI) (T)
- プロンプトにワイルドカードを使用することで、バリエーションを増やすことができます！ (`--prompt_wildcards /path/to/wildcard/directory` のように指定します。例えば、プロンプトで `__color__` と指定すると、そのディレクトリ内の color.txt が検索されます。ワイルドカードファイルの形式は、1行につき1つの置換文字列で、red:2.0 や "some long string:0.5" のように相対的な重みをオプションで付加できます。ワイルドカード自体にワイルドカードを含めることも可能で、再帰回数の制限は50回です。) (HY) (WV) (FP) (FX) (QI)
- シードとして文字列を使用するのは良いでしょう。覚えやすいのも魅力です。 (HY) (WV) (FP) (FX) (QI)
- 推論用の外部LoRAを、事前に変換することなく読み込みます (HY) (WV) (FP) (FX) (QI)
- 決定論を保証する強力な世代ごとのグローバルシード (HY) (WV) (FP) (FX) (QI)
- 生成中にlatent2RGBまたはTAEHV（`--preview_latent_every N`、Nはステップ数（フレームパックの場合はセクション数））を使用した潜在プレビュー。デフォルトではlatent2rgbを使用しますが、TAEは`--preview_vae /path/to/model`で有効にできます。モデル：https://huggingface.co/Blyss/BlissfulModels/tree/main/taehv）(HY) (WV) (FP) (FX)
- 高速で高品質な生成のために最適化された生成設定（`--optimized`\*、モデルに基づいてさまざまな最適化と設定を有効にします。SageAttention、Triton、PyTorch 2.7.0以降が必要です）(HY) (WV) (FP) (FX)
- FP16 積分 (`--fp16_accumulation`、Wan FP16 モデルで最も効果的に機能します (Hunyaun bf16 でも動作します!)。PyTorch 2.7.0 以上が必要ですが、推論速度が大幅に向上します。特に `--compile`\* を使用すると、精度を損なうことなく fp8_fast/mmscaled とほぼ同等の速度を実現できます。また、fp8 スケールモードでも動作します!) (HY) (WV) (FP) (FX)
- 拡張保存オプション (`--codec codec --container container`、Apple ProRes (`--codec prores`、超高ビットレートで知覚的にロスレス) を `--container mkv` に保存、または `h264`、`h265` のいずれかを `mp4` または `mkv` に保存可能) (HY) (WV) (FP)
- 生成メタデータを動画/画像に保存 (自動`--container mkv` を使用し、PNG 保存時は `--no-metadata` で無効にしてください。`--container mp4` では無効です。こうしたメタデータは `src/blissful_tuner/metaview.py some_video.mkv` で簡単に表示/コピーできます。ビューアには mediainfo_cli が必要です) (HY) (WV) (FP) (FX)
- CFGZero* 例: https://github.com/WeichenFan/CFG-Zero-star (`--cfgzerostar_scaling --cfgzerostar_init_steps N` で、N は開始時に 0 になるまでのステップ数です。T2V の場合は 2、I2V の場合は 1 が適切ですが、私の経験では T2V の方が適しています。Hunyuan のサポートは非​​常に実験的であり、CFG が有効になっている場合にのみ利用可能です。) (HY) (WV) (FX)
- 高度な CFG スケジューリング: (`--cfg_schedule`、使用方法については `--help` を参照してください。必要に応じて、個々のステップにガイダンススケールダウンを指定することもできます!) (HY) (WV) (FX)
- RifleX 例:より長い動画の場合は https://github.com/thu-ml/RIFLEx をご覧ください (`--riflex_index N`、N は RifleX の周波数です。Wan の場合は 6 が適しており、通常 81 フレームではなく約 115 フレームまで再生できます。Wan の場合は `--rope_func comfy` が必要です。Hunyuan の場合は 4 が適しており、少なくとも 2 倍の長さにできます!) (HY) (WV)
- 垂直ネガティブガイダンス (`--perp_neg neg_strength`、neg_strength はネガティブプロンプトの文字列を制御する浮動小数点数です。詳しくは `--help` を参照してください!) (HY) (WV)
- 正規化アテンションガイダンス (NAG) (https://arxiv.org/pdf/2505.21179) (クロスアテンション層内でネガティブガイダンスを提供します。通常の CFG だけでなく、蒸留モデルでも動作します。有効にするには`--nag_scale 3.0` を指定して否定プロンプトを表示してください！ (WV)
- 高品質かつ低ステップの蒸留サンプリング（https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill のような蒸留Wanモデル/LoRAで`--sample_solver lcm`または`--sample_solver dpm++sde`を使用します。さらに、便利なLoRAも作成しました: https://huggingface.co/Blyss/BlissfulModels/tree/main/wan_lcm ）(WV)
- V2V推論（`--video_path /path/to/input/video --denoise_strength amount`。amountは0.0～1.0の浮動小数点数で、ソースビデオに追加するノイズの強度を制御します。`--noise_mode Traditional`の場合、他の実装と同様に、タイムステップスケジュールの最後の（amount * 100）パーセントを実行します。`--noise_mode direct`の場合、タイムステップスケジュール内でその値に最も近いところから開始し、そこから処理を進めることで、追加されるノイズの量を可能な限り正確に制御します。スケーリング、パディング、切り捨てをサポートしているため、入力は出力と同じ解像度や長さである必要はありません。`--video_length` が入力より短い場合、入力は切り捨てられ、最初の `--video_length` フレームのみが含まれます。`--video_length` が入力より長い場合、`--v2v_pad_mode` に応じて最初のフレームまたは最後のフレームが繰り返され、長さがパディングされます。T2V または I2V の `--task` モードとモデルを使用できます (i2v モードの方が品質が高いと思います)。I2V モードでは、`--image_path` が指定されていない場合、代わりにビデオの最初のフレームがモデルの調整に使用されます。`--infer_steps` は、完全なノイズ除去の場合と同じ量である必要があります (例: デフォルト)。 T2Vの場合は50、I2Vの場合は40です。これは、フルスケジュールから変更する必要があるためです。実際の手順は`--noise_mode`に依存します。(WV)
- I2I推論 (`--i2i_path /path/to/image` - T2IモードでT2Vモデルを使用する場合、`--denoise_strength`で強度を指定します。潜在ノイズの増強には`--i2_extra_noise`もサポートされています。) (WV)
- プロンプトの重み付け (`--prompt_weighting`を使用し、プロンプトで「(large:1.4)の赤いボールで遊ぶ猫」のように記述することで、「large」の効果を強調できます。[this]や(this)はサポートされておらず、(this:1.0)のみがサポートされています。(WV) (FX)
- 複素数を使用しないComfyUIから移植されたROPE。推論または`--compile`\*と併用すると、VRAMを大幅に節約できます。学習には `--optimized_compile`\* を使用してください！(`--rope_func comfy`) (WV) (T)
- I2V/V2V/I2I 用のオプションの潜在ノイズ (`--v2_extra_noise 0.02 --i2_extra_noise 0.02`、0.04 未満の値を推奨。これにより、細かいディテールやテクスチャが向上しますが、値が大きすぎるとアーティファクトや影の動きが発生します。私は V2V の場合は 0.01～0.02、I2V の場合は 0.02～0.04 程度を使用しています) (WV)
- 混合精度トランスフォーマーをロードします (推論または学習には `--mixed_precision_transformer` を使用します。このようなトランスフォーマーの作成方法と、その理由については https://github.com/kohya-ss/musubi-tuner/discussions/232#discussioncomment-13284677 を参照してください) (WV) (T)
- LLMオプションの追加 (`--hidden_​​state_skip_layer N --apply_final_norm`、説明は`--help`を参照してください!) (HY)
- Wanと同じアルゴリズムを使用したFP8スケールのサポート (`--fp8_scaled`、推論と学習の両方に強く推奨。FP8が優れているだけなので、これだけ知っておく必要があります!) (HY) (T)
- CLIP用のプロンプトの分離 (`--prompt_2 "second prompt goes here"`、CLIPはよりシンプルなテキストに使用されるため、CLIPとは異なるプロンプトを提供します) (HY)
- https://github.com/zer0int/ComfyUI-HunyuanVideo-Nyan に基づいてテキストエンコーダーを再スケール (`--te_multiplier llm clip`、例えば`--te_multiplier 0.9 1.2`のように、LLMの重みをわずかに下げ、CLIPの重みを上げる)（HY）

モデルに依存しない追加機能：
（以下のスクリプトを使用する場合は、`--group postprocess` オプションを使用してプロジェクトを venv にインストールしてください（例：`pip install -e . --group postprocess --group dev` ですべての要件を完全にインストールしてください！）
- GIMM-VFI フレームレート補間（`src/blissful_tuner/GIMMVFI.py`、使用方法については `--help` を参照してください。モデル：https://huggingface.co/Blyss/BlissfulModels/tree/main/VFI）
- SwinIR または ESRGAN タイプのモデルによるアップスケーリング（`src/blissful_tuner/upscaler.py`、使用方法については `--help` を参照してください。モデル：https://huggingface.co/Blyss/BlissfulModels/tree/main/upscaling）
- スクリプトベースの顔ぼかしYolo で - 顔の修正を行わない LoRA のトレーニングに役立ちます！（`blissful_tuner/yolo_blur.py`、使用方法については `--help` を参照してください。推奨モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/yolo ）
- CodeFormer/GFPGAN による顔の修復（`src/blissful_tuner/facefix.py`、いつものように `--help` を参照してください！モデル: https://huggingface.co/Blyss/BlissfulModels/tree/main/face_restoration ）

(\*) - torch.compile に関連する機能には追加の要件があり、ネイティブ Windows プラットフォームでは大きな制限があるため、代わりに WSL2 またはネイティブ Linux 環境をお勧めします。

また、私の関連プロジェクト（ https://github.com/Sarania/Envious ）は、LinuxのターミナルからNvidia GPUを管理するのに便利です。nvidia-ml-pyが必要ですが、リアルタイムモニタリング、オーバークロック/アンダークロック、電力制限調整、ファン制御、プロファイルなどをサポートしています。GPU VRAM用の小さなプロセスモニターも付いています！nvidia-smiがダメな場合のnvidia-smiのようなものです😂

私のコード全体と Musubi Tuner のコードは Apache 2.0 ライセンスです。含まれている他のプロジェクトはライセンスが異なる場合があります。その場合は、それぞれのディレクトリにライセンス条項を記載した LICENSE ファイルがあります。以下は、現在でも有効なオリジナルの Musubi Readme です。
(機械翻訳の終了)

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
    - [ドキュメント](#ドキュメント)
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

- 2026/01/11
    - Qwen-Image-LayeredのLoRA学習に対応しました。[PR #816](https://github.com/kohya-ss/musubi-tuner/pull/816)
        - 詳細は[ドキュメント](./docs/qwen_image.md)を参照してください。
        - キャッシュ作成、学習、推論の各スクリプトで、`--model_version` オプションに `layered` を指定してください。

- 2025/12/27
    - Qwen-Image-Edit-2511に対応しました。[PR #808](https://github.com/kohya-ss/musubi-tuner/pull/808)
        - チェックポイントやオプションの詳細など、詳細は[ドキュメント](./docs/qwen_image.md)を参照してください。
        - キャッシュ作成、学習、推論の各スクリプトで、`--model_version` オプションに `edit-2511` を指定してください。

- 2025/12/25
    - Kandinsky 5のLoRA学習に対応しました。[PR #774](https://github.com/kohya-ss/musubi-tuner/pull/774) AkaneTendo25氏に深く感謝します。
        - 詳細は[ドキュメント](./docs/kandinsky5.md)を参照してください。
        - **重みの指定が一部、Hugging FaceのID形式になっています。近日中に（他のモデルと同様の）*.safetensorsの直接指定方式に変更予定ですのでご注意ください。**

- 2025/12/13
    - Z-Imageのfinetuningに対応しました。[PR #778](https://github.com/kohya-ss/musubi-tuner/pull/778) sdbds氏に深く感謝します。
        - 詳細は[ドキュメント](./docs/zimage.md#finetuning)を参照してください。
    - ごくシンプルなGUIツールを追加しました。[PR #779](https://github.com/kohya-ss/musubi-tuner/pull/779)
        - 現在はZ-Image-TurboとQwen-ImageのLoRA学習に対応しています。詳細は[ドキュメント](./src/musubi_tuner/gui/gui.ja.md)を参照してください。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

### AIコーディングエージェントを使用する開発者の方へ

このリポジトリでは、ClaudeやGeminiのようなAIエージェントが、プロジェクトの概要や構造を理解しやすくするためのエージェント向け文書（プロンプト）を用意しています。

これらを使用するためには、プロジェクトのルートディレクトリに各エージェント向けの設定ファイルを作成し、明示的に読み込む必要があります。

**セットアップ手順:**

1.  プロジェクトのルートに `CLAUDE.md` や `GEMINI.md`、`AGENTS.md` ファイルを作成します。
2.  `CLAUDE.md` 等に以下の行を追加して、リポジトリが推奨するプロンプトをインポートします（現在、両者はほぼ同じ内容です）：

    ```markdown
    @./.ai/claude.prompt.md
    ```

    Geminiの場合はこちらです：

    ```markdown
    @./.ai/gemini.prompt.md
    ```

    他のエージェント向けの設定ファイルでもそれぞれの方法でインポートしてください。

3.  インポートした行の後に、必要な指示を適宜追加してください（例：`Always respond in Japanese.`）。

このアプローチにより、共有されたプロジェクトのコンテキストを活用しつつ、エージェントに与える指示を各ユーザーが自由に制御できます。`CLAUDE.md`、`GEMINI.md` および `AGENTS.md` （またClaude用の `.mcp.json`）はすでに `.gitignore` に記載されているため、リポジトリにコミットされることはありません。

## 概要

### ハードウェア要件

- VRAM: 静止画での学習は12GB以上推奨、動画での学習は24GB以上推奨。
    - *アーキテクチャ、解像度等の学習設定により異なります。*12GBでは解像度 960x544 以下とし、`--blocks_to_swap`、`--fp8_llm`等の省メモリオプションを使用してください。
- メインメモリ: 64GB以上を推奨、32GB+スワップで動作するかもしれませんが、未検証です。

### 特徴

- 省メモリに特化
- Windows対応（Linuxでの動作報告もあります）
- マルチGPU学習（[Accelerate](https://huggingface.co/docs/accelerate/index)を使用）、ドキュメントは後日追加予定

### ドキュメント

各アーキテクチャの詳細、設定、高度な機能については、以下のドキュメントを参照してください。

**アーキテクチャ別:**
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [Wan2.1/2.2 (1フレーム推論)](./docs/wan_1f.md)
- [FramePack](./docs/framepack.md)
- [FramePack (1フレーム推論)](./docs/framepack_1f.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

**共通設定・その他:**
- [データセット設定](./docs/dataset_config.md)
- [高度な設定](./docs/advanced_config.md)
- [学習中のサンプル生成](./docs/sampling_during_training.md)
- [ツールとユーティリティ](./docs/tools.md)
- [torch.compileの使用方法](./docs/torch_compile.md)

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

モデルのダウンロード手順はアーキテクチャによって異なります。詳細は[ドキュメント](#ドキュメント)セクションにある、各アーキテクチャのドキュメントを参照してください。

## 使い方

### データセット設定

[こちら](./docs/dataset_config.md)を参照してください。

### 事前キャッシュ

事前キャッシュの手順の詳細は、[ドキュメント](#ドキュメント)セクションにある各アーキテクチャのドキュメントを参照してください。

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

学習と推論の手順はアーキテクチャによって大きく異なります。詳細な手順については、[ドキュメント](#ドキュメント)セクションにある対応するアーキテクチャのドキュメント、および各種の設定のドキュメントを参照してください。

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

コントリビューションを歓迎します。 [CONTRIBUTING.md](./CONTRIBUTING.md)および[CONTRIBUTING.ja.md](./CONTRIBUTING.ja.md)をご覧ください。

## ライセンス

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

`wan`ディレクトリ以下のコードは、[Wan2.1](https://github.com/Wan-Video/Wan2.1)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

`frame_pack`ディレクトリ以下のコードは、[frame_pack](https://github.com/lllyasviel/FramePack)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
