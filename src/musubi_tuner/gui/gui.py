import gradio as gr
import os
import toml
from musubi_tuner.gui.config_manager import ConfigManager

config_manager = ConfigManager()

# UI Text Dictionary for potential i18n
# I18N Configuration
I18N_DATA = {
    "en": {
        "app_title": "Musubi Tuner GUI",
        "app_header": "# Musubi Tuner GUI",
        "app_desc": "A simple frontend for training LoRA models with Musubi Tuner.",
        "acc_project": "1. Project Settings",
        "desc_project": "All working files will be created under this directory.",
        "lbl_proj_dir": "Project Working Directory",
        "ph_proj_dir": "Absolute path to your project folder",
        "btn_init_project": "Initialize/Load Project",
        "acc_model": "2. Model & Dataset Configuration",
        "desc_model": "Choose the model architecture and specify the ComfyUI models directory.",
        "lbl_model_arch": "Model Architecture",
        "lbl_vram": "VRAM Size (GB)",
        "lbl_comfy_dir": "ComfyUI Models Directory",
        "ph_comfy_dir": "Absolute path to ComfyUI/models",
        "btn_validate_models": "Validate Models Directory",
        "header_dataset": "### 3. Dataset Settings",
        "desc_dataset": "Configure the resolution and batch size for the dataset.",
        "btn_rec_res_batch": "Set Recommended Resolution & Batch Size",
        "lbl_res_w": "Resolution Width",
        "lbl_res_h": "Resolution Height",
        "lbl_batch_size": "Batch Size",
        "btn_gen_config": "Generate Dataset Config",
        "lbl_toml_preview": "TOML Preview",
        "acc_preprocessing": "4. Preprocessing",
        "desc_preprocessing": "Pre-calculate latents and text encoder outputs to speed up training.",
        "btn_set_paths": "Set Default Paths",
        "lbl_vae_path": "VAE Path",
        "ph_vae_path": "Path to VAE model",
        "lbl_te1_path": "Text Encoder 1 Path",
        "ph_te1_path": "Path to Text Encoder 1",
        "lbl_te2_path": "Text Encoder 2 Path",
        "ph_te2_path": "Path to Text Encoder 2 (Optional)",
        "btn_cache_latents": "Cache Latents",
        "btn_cache_text": "Cache Text Encoder Outputs",
        "lbl_cache_log": "Caching Log Output",
        "acc_training": "5. Training",
        "desc_training_basic": "Configure the training parameters.",
        "desc_training_zimage": "Recommended: Use **bf16** for mixed precision. Z-Image requires specific attention to Flow Shift.",
        "btn_rec_params": "Set Recommended Parameters",
        "lbl_dit_path": "Base Model / DiT Path",
        "ph_dit_path": "Path to DiT model",
        "lbl_output_name": "Output LoRA Name",
        "header_basic_params": "### Basic Parameters",
        "lbl_lr": "Learning Rate",
        "lbl_epochs": "Epochs",
        "lbl_save_every": "Save Every N Epochs",
        "accordion_advanced": "Advanced Parameters",
        "desc_training_detailed": """
### detailed explanation
- **Learning Rate**: Controls how much the model weights are updated during training. Lower values are safer but slower.
- **Epochs**: One complete pass through the entire training dataset.
- **Save Every N Epochs**: How often to save the model and generate sample images.
- **Discrete Flow Shift**: A parameter specific to flow matching models. 2.0 is recommended for Z-Image.
- **Block Swap**: Offloads model blocks to CPU to save VRAM. Higher values save more VRAM but slow down training.
- **Mixed Precision**: bf16 is recommended for modern GPUs (RTX 30xx+). fp16 uses less memory but is less stable.
- **Gradient Checkpointing**: Saves VRAM by recomputing activations during backward pass.
- **FP8**: further reduces memory usage by using 8-bit floating point arithmetic.
""",
        "lbl_flow_shift": "Discrete Flow Shift",
        "lbl_block_swap": "Block Swap (0-28)",
        "lbl_mixed_precision": "Mixed Precision",
        "lbl_grad_cp": "Gradient Checkpointing",
        "lbl_fp8_scaled": "FP8 Scaled (DiT) - Enables --fp8_base and --fp8_scaled",
        "lbl_fp8_llm": "FP8 LLM (Text Encoder)",
        "accordion_additional": "Additional Options",
        "desc_additional_args": "Enter any additional command line arguments here. They will be appended to the training command.",
        "lbl_additional_args": "Additional Optional Arguments",
        "ph_additional_args": "--arg value --flag",
        "btn_start_training": "Start Training (New Window)",
        "acc_post_processing": "6. Post-Processing",
        "desc_post_proc": "Convert Z-Image LoRA to ComfyUI format.",
        "lbl_input_lora": "Input LoRA Path",
        "ph_input_lora": "Path to trained .safetensors file",
        "lbl_output_comfy": "Output ComfyUI LoRA Path",
        "ph_output_comfy": "Path to save converted model",
        "btn_convert": "Convert to ComfyUI Format",
        "lbl_conversion_log": "Conversion Log",
        "desc_qwen_notes": "Qwen-Image specific notes here.",
    },
    "ja": {
        "app_title": "Musubi Tuner GUI",
        "app_header": "# Musubi Tuner GUI",
        "app_desc": "Musubi TunerでLoRAモデルを学習するためのシンプルなフロントエンドです。",
        "acc_project": "1. プロジェクト設定",
        "desc_project": "すべての作業ファイルはこのディレクトリ下に作成されます。",
        "lbl_proj_dir": "プロジェクト作業ディレクトリ",
        "ph_proj_dir": "プロジェクトフォルダへの絶対パス",
        "btn_init_project": "プロジェクトを初期化/読み込み",
        "acc_model": "2. モデル＆データセット設定",
        "desc_model": "モデルアーキテクチャを選択し、ComfyUIのモデルディレクトリを指定してください。",
        "lbl_model_arch": "モデルアーキテクチャ",
        "lbl_vram": "VRAMサイズ (GB)",
        "lbl_comfy_dir": "ComfyUI モデルディレクトリ",
        "ph_comfy_dir": "ComfyUI/models への絶対パス",
        "btn_validate_models": "モデルディレクトリを検証",
        "header_dataset": "### 3. データセット設定",
        "desc_dataset": "データセットの解像度とバッチサイズを設定してください。",
        "btn_rec_res_batch": "推奨解像度とバッチサイズを設定",
        "lbl_res_w": "解像度 幅",
        "lbl_res_h": "解像度 高さ",
        "lbl_batch_size": "バッチサイズ",
        "btn_gen_config": "データセット設定(TOML)を生成",
        "lbl_toml_preview": "TOML プレビュー",
        "acc_preprocessing": "4. 前処理 (Preprocessing)",
        "desc_preprocessing": "学習を高速化するためにLatentsとテキストエンコーダーの出力を事前計算します。",
        "btn_set_paths": "デフォルトパスを設定",
        "lbl_vae_path": "VAE パス",
        "ph_vae_path": "VAEモデルへのパス",
        "lbl_te1_path": "テキストエンコーダー1 パス",
        "ph_te1_path": "テキストエンコーダー1へのパス",
        "lbl_te2_path": "テキストエンコーダー2 パス",
        "ph_te2_path": "テキストエンコーダー2へのパス (オプション)",
        "btn_cache_latents": "Latentsをキャッシュ",
        "btn_cache_text": "テキストエンコーダー出力をキャッシュ",
        "lbl_cache_log": "キャッシュログ出力",
        "acc_training": "5. 学習 (Training)",
        "desc_training_basic": "学習パラメータを設定してください。",
        "desc_training_zimage": "推奨: 混合精度には **bf16** を使用してください。Z-ImageはFlow Shiftに注意が必要です。",
        "btn_rec_params": "推奨パラメータを設定",
        "lbl_dit_path": "ベースモデル / DiT パス",
        "ph_dit_path": "DiTモデルへのパス",
        "lbl_output_name": "出力 LoRA 名",
        "header_basic_params": "### 基本パラメータ",
        "lbl_lr": "学習率 (Learning Rate)",
        "lbl_epochs": "エポック数 (Epochs)",
        "lbl_save_every": "Nエポックごとに保存",
        "accordion_advanced": "詳細パラメータ",
        "desc_training_detailed": """
### 詳細説明
- **学習率 (Learning Rate)**: 学習中にモデルの重みをどれくらい更新するかを制御します。低い値の方が安全ですが、学習が遅くなります。
- **エポック数 (Epochs)**: 学習データセット全体を通す回数です。
- **保存頻度 (Save Every N Epochs)**: モデルの保存とサンプル生成を行う頻度です。
- **Discrete Flow Shift**: Flow Matchingモデル特有のパラメータです。Z-Imageでは2.0が推奨されます。
- **Block Swap**: VRAMを節約するためにモデルブロックをCPUにオフロードします。値を大きくするとVRAMを節約できますが、学習が遅くなります。
- **混合精度 (Mixed Precision)**: 最新のGPU(RTX 30xx以降)ではbf16が推奨されます。fp16はメモリ使用量が少ないですが、安定性が低いです。
- **Gradient Checkpointing**: Backwardパス中にアクティベーションを再計算することでVRAMを節約します。
- **FP8**: 8ビット浮動小数点演算を使用することでメモリ使用量をさらに削減します。
""",
        "lbl_flow_shift": "Discrete Flow Shift",
        "lbl_block_swap": "Block Swap (0-28)",
        "lbl_mixed_precision": "混合精度 (Mixed Precision)",
        "lbl_grad_cp": "Gradient Checkpointing",
        "lbl_fp8_scaled": "FP8 Scaled (DiT) - --fp8_base と --fp8_scaled を有効化",
        "lbl_fp8_llm": "FP8 LLM (テキストエンコーダー)",
        "accordion_additional": "追加オプション",
        "desc_additional_args": "追加のコマンドライン引数を入力してください。これらは学習コマンドに追加されます。",
        "lbl_additional_args": "追加のオプション引数",
        "ph_additional_args": "--arg value --flag",
        "btn_start_training": "学習を開始 (新しいウィンドウ)",
        "acc_post_processing": "6. 後処理 (Post-Processing)",
        "desc_post_proc": "Z-Image LoRAをComfyUI形式に変換します。",
        "lbl_input_lora": "入力 LoRA パス",
        "ph_input_lora": "学習済み .safetensors ファイルへのパス",
        "lbl_output_comfy": "出力 ComfyUI LoRA パス",
        "ph_output_comfy": "変換後のモデルの保存先パス",
        "btn_convert": "ComfyUI形式に変換",
        "lbl_conversion_log": "変換ログ",
        "desc_qwen_notes": "Qwen-Image 特有の注意点。",
    }
}

i18n = gr.I18n(en=I18N_DATA["en"], ja=I18N_DATA["ja"])


def construct_ui():
    with gr.Blocks(title=i18n("app_title")) as demo:
        gr.Markdown(i18n("app_header"))
        gr.Markdown(i18n("app_desc"))

        with gr.Accordion(i18n("acc_project"), open=True):
            gr.Markdown(i18n("desc_project"))
            with gr.Row():
                project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"))

            # Placeholder for project initialization or loading
            init_btn = gr.Button(i18n("btn_init_project"))
            project_status = gr.Markdown("")

            # Old init_project removed, replaced by logic in 'Dataset Settings' block to handle loading settings.

        with gr.Accordion(i18n("acc_model"), open=True):
            gr.Markdown(i18n("desc_model"))
            with gr.Row():
                model_arch = gr.Dropdown(
                    label=i18n("lbl_model_arch"),
                    choices=[
                        "Z-Image",
                        "Qwen-Image",
                    ],
                    value="Z-Image",
                )
                vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24")

            with gr.Row():
                comfy_models_dir = gr.Textbox(label=i18n("lbl_comfy_dir"), placeholder=i18n("ph_comfy_dir"))

            # Validation for ComfyUI models directory
            models_status = gr.Markdown("")
            validate_models_btn = gr.Button(i18n("btn_validate_models"))

            # Placeholder for Dataset Settings (Step 3)
            gr.Markdown(i18n("header_dataset"))
            gr.Markdown(i18n("desc_dataset"))
            with gr.Row():
                set_rec_settings_btn = gr.Button(i18n("btn_rec_res_batch"))
            with gr.Row():
                resolution_w = gr.Number(label=i18n("lbl_res_w"), value=1024, precision=0)
                resolution_h = gr.Number(label=i18n("lbl_res_h"), value=1024, precision=0)
                batch_size = gr.Number(label=i18n("lbl_batch_size"), value=1, precision=0)

            gen_toml_btn = gr.Button(i18n("btn_gen_config"))
            dataset_status = gr.Markdown("")
            toml_preview = gr.Code(label=i18n("lbl_toml_preview"), interactive=False)

            def load_project_settings(project_path):
                settings = {}
                try:
                    settings_path = os.path.join(project_path, "musubi_project.toml")
                    if os.path.exists(settings_path):
                        with open(settings_path, "r", encoding="utf-8") as f:
                            settings = toml.load(f)
                except Exception as e:
                    print(f"Error loading project settings: {e}")
                return settings

            def load_dataset_config_content(project_path):
                content = ""
                try:
                    config_path = os.path.join(project_path, "dataset_config.toml")
                    if os.path.exists(config_path):
                        with open(config_path, "r", encoding="utf-8") as f:
                            content = f.read()
                except Exception as e:
                    print(f"Error reading dataset config: {e}")
                return content

            def save_project_settings(project_path, **kwargs):
                try:
                    # Load existing settings to support partial updates
                    settings = load_project_settings(project_path)
                    # Update with new values
                    settings.update(kwargs)

                    settings_path = os.path.join(project_path, "musubi_project.toml")
                    with open(settings_path, "w", encoding="utf-8") as f:
                        toml.dump(settings, f)
                except Exception as e:
                    print(f"Error saving project settings: {e}")

            def init_project(path):
                if not path:
                    return (
                        "Please enter a project directory path.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    os.makedirs(os.path.join(path, "training"), exist_ok=True)

                    # Load settings if available
                    settings = load_project_settings(path)
                    new_model = settings.get("model_arch", "Z-Image")
                    new_vram = settings.get("vram_size", "24")
                    new_comfy = settings.get("comfy_models_dir", "")
                    new_w = settings.get("resolution_w", 1024)
                    new_h = settings.get("resolution_h", 1024)
                    new_batch = settings.get("batch_size", 1)
                    new_vae = settings.get("vae_path", "")
                    new_te1 = settings.get("text_encoder1_path", "")
                    new_te2 = settings.get("text_encoder2_path", "")

                    # Training params
                    new_dit = settings.get("dit_path", "")
                    new_out_nm = settings.get("output_name", "my_lora")
                    new_lr = settings.get("learning_rate", 1e-4)
                    new_epochs = settings.get("num_epochs", 16)
                    new_save_n = settings.get("save_every_n_epochs", 1)
                    new_flow = settings.get("discrete_flow_shift", 2.0)
                    new_swap = settings.get("block_swap", 0)
                    new_prec = settings.get("mixed_precision", "bf16")
                    new_grad_cp = settings.get("gradient_checkpointing", True)
                    new_fp8_s = settings.get("fp8_scaled", True)
                    new_fp8_l = settings.get("fp8_llm", True)
                    new_add_args = settings.get("additional_args", "")

                    # Post-processing params
                    new_in_lora = settings.get("input_lora_path", "")
                    new_out_comfy = settings.get("output_comfy_lora_path", "")

                    # Load dataset config content
                    preview_content = load_dataset_config_content(path)

                    msg = f"Project initialized at {path}. 'training' folder ready."
                    if settings:
                        msg += " Settings loaded."

                    return (
                        msg,
                        new_model,
                        new_vram,
                        new_comfy,
                        new_w,
                        new_h,
                        new_batch,
                        preview_content,
                        new_vae,
                        new_te1,
                        new_te2,
                        new_dit,
                        new_out_nm,
                        new_lr,
                        new_epochs,
                        new_save_n,
                        new_flow,
                        new_swap,
                        new_prec,
                        new_grad_cp,
                        new_fp8_s,
                        new_fp8_l,
                        new_add_args,
                        new_in_lora,
                        new_out_comfy,
                    )
                except Exception as e:
                    return (
                        f"Error initializing project: {str(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            def generate_config(project_path, w, h, batch, model_val, vram_val, comfy_val, vae_val, te1_val, te2_val):
                if not project_path:
                    return "Error: Project directory not specified.", ""

                # Save project settings first
                save_project_settings(
                    project_path,
                    model_arch=model_val,
                    vram_size=vram_val,
                    comfy_models_dir=comfy_val,
                    resolution_w=w,
                    resolution_h=h,
                    batch_size=batch,
                    vae_path=vae_val,
                    text_encoder1_path=te1_val,
                    text_encoder2_path=te2_val,
                )

                # Normalize paths
                project_path = os.path.abspath(project_path)
                image_dir = os.path.join(project_path, "training").replace("\\", "/")
                cache_dir = os.path.join(project_path, "cache").replace("\\", "/")

                toml_content = f"""# Auto-generated by Musubi Tuner GUI

[general]
resolution = [{int(w)}, {int(h)}]
caption_extension = ".txt"
batch_size = {int(batch)}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
num_repeats = 1
"""
                try:
                    config_path = os.path.join(project_path, "dataset_config.toml")
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(toml_content)
                    return f"Successfully generated config at {config_path}", toml_content
                except Exception as e:
                    return f"Error generating config: {str(e)}", ""

        with gr.Accordion(i18n("acc_preprocessing"), open=False):
            gr.Markdown(i18n("desc_preprocessing"))
            with gr.Row():
                set_preprocessing_defaults_btn = gr.Button(i18n("btn_set_paths"))
            with gr.Row():
                vae_path = gr.Textbox(label=i18n("lbl_vae_path"), placeholder=i18n("ph_vae_path"))
                text_encoder1_path = gr.Textbox(label=i18n("lbl_te1_path"), placeholder=i18n("ph_te1_path"))
                text_encoder2_path = gr.Textbox(label=i18n("lbl_te2_path"), placeholder=i18n("ph_te2_path"))

            with gr.Row():
                cache_latents_btn = gr.Button(i18n("btn_cache_latents"))
                cache_text_btn = gr.Button(i18n("btn_cache_text"))

            # Simple output area for caching logs
            caching_output = gr.Textbox(label=i18n("lbl_cache_log"), lines=10, interactive=False)

            def validate_models_dir(path):
                if not path:
                    return "Please enter a ComfyUI models directory."

                required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                missing = []
                for d in required_subdirs:
                    if not os.path.join(path, d) or not os.path.exists(os.path.join(path, d)):
                        missing.append(d)

                if missing:
                    return f"Error: Missing subdirectories in models folder: {', '.join(missing)}"

                return "Valid ComfyUI models directory structure found."

            def set_recommended_settings(project_path, model_arch, vram_val):
                w, h = config_manager.get_resolution(model_arch)
                chk_batch = config_manager.get_batch_size(model_arch, vram_val)

                if project_path:
                    save_project_settings(project_path, resolution_w=w, resolution_h=h, batch_size=chk_batch)
                return w, h, chk_batch

            def set_preprocessing_defaults(project_path, comfy_models_dir, model_arch):
                if not comfy_models_dir:
                    return gr.update(), gr.update(), gr.update()

                vae_default, te1_default, te2_default = config_manager.get_preprocessing_paths(model_arch, comfy_models_dir)
                if not te2_default:
                    te2_default = ""  # Ensure empty string for text input

                if project_path:
                    save_project_settings(
                        project_path, vae_path=vae_default, text_encoder1_path=te1_default, text_encoder2_path=te2_default
                    )

                return vae_default, te1_default, te2_default

            def set_training_defaults(project_path, comfy_models_dir, model_arch, vram_val):
                defaults = config_manager.get_training_defaults(model_arch, vram_val, comfy_models_dir)

                dit_default = defaults.get("dit_path", "")
                lr = defaults.get("learning_rate", 1e-4)
                epochs = defaults.get("num_epochs", 16)
                save_n = defaults.get("save_every_n_epochs", 1)
                flow = defaults.get("discrete_flow_shift", 2.0)
                swap = defaults.get("block_swap", 0)
                prec = defaults.get("mixed_precision", "bf16")
                grad_cp = defaults.get("gradient_checkpointing", True)
                fp8_s = defaults.get("fp8_scaled", True)
                fp8_l = defaults.get("fp8_llm", True)

                if project_path:
                    save_project_settings(
                        project_path,
                        dit_path=dit_default,
                        learning_rate=lr,
                        num_epochs=epochs,
                        save_every_n_epochs=save_n,
                        discrete_flow_shift=flow,
                        block_swap=swap,
                        mixed_precision=prec,
                        gradient_checkpointing=grad_cp,
                        fp8_scaled=fp8_s,
                        fp8_llm=fp8_l,
                        vram_size=vram_val,  # Ensure VRAM size is saved
                    )

                return dit_default, lr, epochs, save_n, flow, swap, prec, grad_cp, fp8_s, fp8_l

            def set_post_processing_defaults(project_path, output_nm):
                if not project_path or not output_nm:
                    return gr.update(), gr.update()

                models_dir = os.path.join(project_path, "models")
                in_lora = os.path.join(models_dir, f"{output_nm}.safetensors")
                out_lora = os.path.join(models_dir, f"{output_nm}_comfy.safetensors")

                save_project_settings(project_path, input_lora_path=in_lora, output_comfy_lora_path=out_lora)

                return in_lora, out_lora

            import subprocess
            import sys

            def run_command(command):
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        shell=True,
                        text=True,
                        encoding="utf-8",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )

                    output_log = ""
                    for line in process.stdout:
                        output_log += line
                        yield output_log

                    process.wait()
                    if process.returncode != 0:
                        output_log += f"\nError: Process exited with code {process.returncode}"
                        yield output_log

                except Exception as e:
                    yield f"Error executing command: {str(e)}"

            def cache_latents(project_path, vae_path_val, te1, te2, model, comfy, w, h, batch):
                if not project_path:
                    yield "Error: Project directory not set."
                    return

                # Save settings first
                save_project_settings(
                    project_path,
                    model_arch=model,
                    comfy_models_dir=comfy,
                    resolution_w=w,
                    resolution_h=h,
                    batch_size=batch,
                    vae_path=vae_path_val,
                    text_encoder1_path=te1,
                    text_encoder2_path=te2,
                )

                if not vae_path_val:
                    yield "Error: VAE path not set."
                    return

                if not os.path.exists(vae_path_val):
                    yield f"Error: VAE model not found at {vae_path_val}"
                    return

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first."
                    return

                script_path = os.path.join("src", "musubi_tuner", "zimage_cache_latents.py")

                cmd = [sys.executable, script_path, "--dataset_config", config_path, "--vae", vae_path_val]

                command_str = " ".join(cmd)
                yield f"Starting Latent Caching...\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

            def cache_text_encoder(project_path, te1_path_val, te2_path_val, vae, model, comfy, w, h, batch):
                if not project_path:
                    yield "Error: Project directory not set."
                    return

                # Save settings first
                save_project_settings(
                    project_path,
                    model_arch=model,
                    comfy_models_dir=comfy,
                    resolution_w=w,
                    resolution_h=h,
                    batch_size=batch,
                    vae_path=vae,
                    text_encoder1_path=te1_path_val,
                    text_encoder2_path=te2_path_val,
                )

                if not te1_path_val:
                    yield "Error: Text Encoder 1 path not set."
                    return

                if not os.path.exists(te1_path_val):
                    yield f"Error: Text Encoder 1 model not found at {te1_path_val}"
                    return

                # Z-Image only uses te1 for now, but keeping te2 in signature if needed later or for other models

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first."
                    return

                script_path = os.path.join("src", "musubi_tuner", "zimage_cache_text_encoder_outputs.py")

                cmd = [
                    sys.executable,
                    script_path,
                    "--dataset_config",
                    config_path,
                    "--text_encoder",
                    te1_path_val,
                    "--batch_size",
                    "1",  # Conservative default
                ]

                command_str = " ".join(cmd)
                yield f"Starting Text Encoder Caching...\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

        with gr.Accordion(i18n("acc_training"), open=False):
            gr.Markdown(i18n("desc_training_basic"))
            training_model_info = gr.Markdown(i18n("desc_training_zimage"))

            with gr.Row():
                set_training_defaults_btn = gr.Button(i18n("btn_rec_params"))
            with gr.Row():
                dit_path = gr.Textbox(label=i18n("lbl_dit_path"), placeholder=i18n("ph_dit_path"))

            with gr.Row():
                output_name = gr.Textbox(label=i18n("lbl_output_name"), value="my_lora")

            with gr.Group():
                gr.Markdown(i18n("header_basic_params"))
                with gr.Row():
                    learning_rate = gr.Number(label=i18n("lbl_lr"), value=1e-4)
                    num_epochs = gr.Number(label=i18n("lbl_epochs"), value=16)
                    save_every_n_epochs = gr.Number(label=i18n("lbl_save_every"), value=1)

            with gr.Group():
                with gr.Accordion(i18n("accordion_advanced"), open=False):
                    gr.Markdown(i18n("desc_training_detailed"))
                    with gr.Row():
                        discrete_flow_shift = gr.Number(label=i18n("lbl_flow_shift"), value=2.0)
                    block_swap = gr.Slider(label=i18n("lbl_block_swap"), minimum=0, maximum=28, step=1, value=0)

                with gr.Row():
                    mixed_precision = gr.Dropdown(label=i18n("lbl_mixed_precision"), choices=["bf16", "fp16", "no"], value="bf16")
                    gradient_checkpointing = gr.Checkbox(label=i18n("lbl_grad_cp"), value=True)

                with gr.Row():
                    fp8_scaled = gr.Checkbox(label=i18n("lbl_fp8_scaled"), value=True)
                    fp8_llm = gr.Checkbox(label=i18n("lbl_fp8_llm"), value=True)

            with gr.Accordion(i18n("accordion_additional"), open=False):
                gr.Markdown(i18n("desc_additional_args"))
                additional_args = gr.Textbox(label=i18n("lbl_additional_args"), placeholder=i18n("ph_additional_args"))

            training_status = gr.Markdown("")
            start_training_btn = gr.Button(i18n("btn_start_training"), variant="primary")

        with gr.Accordion(i18n("acc_post_processing"), open=False):
            gr.Markdown(i18n("desc_post_proc"))
            with gr.Row():
                set_post_proc_defaults_btn = gr.Button(i18n("btn_set_paths"))
            with gr.Row():
                input_lora = gr.Textbox(label=i18n("lbl_input_lora"), placeholder=i18n("ph_input_lora"))
                output_comfy_lora = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder=i18n("ph_output_comfy"))

            convert_btn = gr.Button(i18n("btn_convert"))
            conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)

        def convert_lora_to_comfy(project_path, input_path, output_path, model, comfy, w, h, batch, vae, te1, te2):
            if not project_path:
                yield "Error: Project directory not set."
                return

            # Save settings
            save_project_settings(
                project_path,
                model_arch=model,
                comfy_models_dir=comfy,
                resolution_w=w,
                resolution_h=h,
                batch_size=batch,
                vae_path=vae,
                text_encoder1_path=te1,
                text_encoder2_path=te2,
                input_lora_path=input_path,
                output_comfy_lora_path=output_path,
            )

            if not input_path or not output_path:
                yield "Error: Input and Output paths must be specified."
                return

            if not os.path.exists(input_path):
                yield f"Error: Input file not found at {input_path}"
                return

            # Script path
            script_path = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
            if not os.path.exists(script_path):
                yield f"Error: Conversion script not found at {script_path}"
                return

            cmd = [sys.executable, script_path, input_path, output_path]

            command_str = " ".join(cmd)
            yield f"Starting Conversion...\nCommand: {command_str}\n\n"

            yield from run_command(command_str)

        def start_training(
            project_path,
            model,
            dit,
            vae,
            te1,
            output_nm,
            lr,
            epochs,
            save_n,
            flow_shift,
            swap,
            prec,
            grad_cp,
            fp8_s,
            fp8_l,
            add_args,
        ):
            import shlex

            if not project_path:
                return "Error: Project directory not set."
            if not dit:
                return "Error: Base Model / DiT Path not set."
            if not vae:
                return "Error: VAE Path not set (configure in Preprocessing)."
            if not te1:
                return "Error: Text Encoder 1 Path not set (configure in Preprocessing)."

            dataset_config = os.path.join(project_path, "dataset_config.toml")
            if not os.path.exists(dataset_config):
                return "Error: dataset_config.toml not found. Please generate it."

            output_dir = os.path.join(project_path, "models")

            # Save settings
            save_project_settings(
                project_path,
                dit_path=dit,
                output_name=output_nm,
                learning_rate=lr,
                num_epochs=epochs,
                save_every_n_epochs=save_n,
                discrete_flow_shift=flow_shift,
                block_swap=swap,
                mixed_precision=prec,
                gradient_checkpointing=grad_cp,
                fp8_scaled=fp8_s,
                fp8_llm=fp8_l,
                vae_path=vae,
                text_encoder1_path=te1,
                additional_args=add_args,
            )

            # Construct command for cmd /c to run and then pause
            # We assume 'accelerate' is in the PATH.
            script_path = os.path.join("src", "musubi_tuner", "zimage_train_network.py")

            # Inner command list - arguments for accelerate launch
            inner_cmd = [
                "accelerate",
                "launch",
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                prec,
                script_path,
                "--dit",
                dit,
                "--vae",
                vae,
                "--text_encoder",
                te1,
                "--dataset_config",
                dataset_config,
                "--output_dir",
                output_dir,
                "--output_name",
                output_nm,
                "--network_module",
                "networks.lora_zimage",
                "--network_dim",
                "32",
                "--optimizer_type",
                "adamw8bit",
                "--learning_rate",
                str(lr),
                "--max_train_epochs",
                str(int(epochs)),
                "--save_every_n_epochs",
                str(int(save_n)),
                "--timestep_sampling",
                "shift",
                "--weighting_scheme",
                "none",
                "--discrete_flow_shift",
                str(flow_shift),
                "--max_data_loader_n_workers",
                "2",
                "--persistent_data_loader_workers",
                "--seed",
                "42",
            ]

            if prec != "no":
                inner_cmd.extend(["--mixed_precision", prec])

            if grad_cp:
                inner_cmd.append("--gradient_checkpointing")

            if fp8_s:
                inner_cmd.append("--fp8_base")
                inner_cmd.append("--fp8_scaled")

            if fp8_l:
                inner_cmd.append("--fp8_llm")

            if swap > 0:
                inner_cmd.extend(["--blocks_to_swap", str(int(swap))])

            inner_cmd.append("--sdpa")

            # Model specific command modification
            if model == "Z-Image":
                pass
            elif model == "Qwen-Image":
                pass

            # Parse and append additional args
            if add_args:
                try:
                    split_args = shlex.split(add_args)
                    inner_cmd.extend(split_args)
                except Exception as e:
                    return f"Error parsing additional arguments: {str(e)}"

            # Construct the full command string for cmd /c
            # list2cmdline will quote arguments as needed for Windows
            inner_cmd_str = subprocess.list2cmdline(inner_cmd)

            # Chain commands: Run training -> echo message -> pause >nul (hides default message)
            final_cmd_str = f"{inner_cmd_str} & echo. & echo Training finished. Press any key to close this window... & pause >nul"

            try:
                # Open in new console window
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                # Pass explicit 'cmd', '/c', string to ensure proper execution
                subprocess.Popen(["cmd", "/c", final_cmd_str], creationflags=flags, shell=False)
                return f"Training started in a new window!\n\nCommand: {inner_cmd_str}"
            except Exception as e:
                return f"Error starting training: {str(e)}"

        def update_model_info(model):
            if model == "Z-Image":
                return i18n("desc_training_zimage")
            elif model == "Qwen-Image":
                return i18n("desc_qwen_notes")
            return ""

        # Event wiring moved to end to prevent UnboundLocalError
        init_btn.click(
            fn=init_project,
            inputs=[project_dir],
            outputs=[
                project_status,
                model_arch,
                vram_size,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                toml_preview,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                dit_path,
                output_name,
                learning_rate,
                num_epochs,
                save_every_n_epochs,
                discrete_flow_shift,
                block_swap,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
                additional_args,
                input_lora,
                output_comfy_lora,
            ],
        )

        model_arch.change(fn=update_model_info, inputs=[model_arch], outputs=[training_model_info])

        gen_toml_btn.click(
            fn=generate_config,
            inputs=[
                project_dir,
                resolution_w,
                resolution_h,
                batch_size,
                model_arch,
                vram_size,
                comfy_models_dir,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
            ],
            outputs=[dataset_status, toml_preview],
        )

        validate_models_btn.click(fn=validate_models_dir, inputs=[comfy_models_dir], outputs=[models_status])

        set_rec_settings_btn.click(
            fn=set_recommended_settings,
            inputs=[project_dir, model_arch, vram_size],
            outputs=[resolution_w, resolution_h, batch_size],
        )

        set_preprocessing_defaults_btn.click(
            fn=set_preprocessing_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch],
            outputs=[vae_path, text_encoder1_path, text_encoder2_path],
        )

        set_post_proc_defaults_btn.click(
            fn=set_post_processing_defaults, inputs=[project_dir, output_name], outputs=[input_lora, output_comfy_lora]
        )

        set_training_defaults_btn.click(
            fn=set_training_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch, vram_size],
            outputs=[
                dit_path,
                learning_rate,
                num_epochs,
                save_every_n_epochs,
                discrete_flow_shift,
                block_swap,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
            ],
        )

        cache_latents_btn.click(
            fn=cache_latents,
            inputs=[
                project_dir,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
            ],
            outputs=[caching_output],
        )

        cache_text_btn.click(
            fn=cache_text_encoder,
            inputs=[
                project_dir,
                text_encoder1_path,
                text_encoder2_path,
                vae_path,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
            ],
            outputs=[caching_output],
        )

        start_training_btn.click(
            fn=start_training,
            inputs=[
                project_dir,
                model_arch,
                dit_path,
                vae_path,
                text_encoder1_path,
                output_name,
                learning_rate,
                num_epochs,
                save_every_n_epochs,
                discrete_flow_shift,
                block_swap,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
                additional_args,
            ],
            outputs=[training_status],
        )

        convert_btn.click(
            fn=convert_lora_to_comfy,
            inputs=[
                project_dir,
                input_lora,
                output_comfy_lora,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
            ],
            outputs=[conversion_log],
        )

    return demo


if __name__ == "__main__":
    demo = construct_ui()
    demo.launch(i18n=i18n)
