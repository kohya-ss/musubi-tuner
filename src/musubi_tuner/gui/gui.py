import glob
import gradio as gr
import os
import toml
from musubi_tuner.gui.config_manager import ConfigManager
from musubi_tuner.gui.i18n_data import I18N_DATA

config_manager = ConfigManager()


i18n = gr.I18n(en=I18N_DATA["en"], ja=I18N_DATA["ja"])


def construct_ui():
    # I18N doesn't work for gr.Blocks title
    # with gr.Blocks(title=i18n("app_title")) as demo:
    with gr.Blocks(title="Musubi Tuner GUI") as demo:
        gr.Markdown(i18n("app_header"))
        gr.Markdown(i18n("app_desc"))

        with gr.Accordion(i18n("acc_project"), open=True):
            gr.Markdown(i18n("desc_project"))
            with gr.Row():
                project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"), max_lines=1)

            # Placeholder for project initialization or loading
            init_btn = gr.Button(i18n("btn_init_project"))
            project_status = gr.Markdown("")

        with gr.Accordion(i18n("acc_model"), open=False):
            gr.Markdown(i18n("desc_model"))
            with gr.Row():
                model_arch = gr.Dropdown(
                    label=i18n("lbl_model_arch"),
                    choices=[
                        "Qwen-Image",
                        "Z-Image-Turbo",
                    ],
                    value="Qwen-Image",
                )
                vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24")

            with gr.Row():
                comfy_models_dir = gr.Textbox(label=i18n("lbl_comfy_dir"), placeholder=i18n("ph_comfy_dir"), max_lines=1)

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
                    new_model = settings.get("model_arch", "Qwen-Image")
                    new_vram = settings.get("vram_size", "16")
                    new_comfy = settings.get("comfy_models_dir", "")
                    new_w = settings.get("resolution_w", 1328)
                    new_h = settings.get("resolution_h", 1328)
                    new_batch = settings.get("batch_size", 1)
                    new_vae = settings.get("vae_path", "")
                    new_te1 = settings.get("text_encoder1_path", "")
                    new_te2 = settings.get("text_encoder2_path", "")

                    # Training params
                    new_dit = settings.get("dit_path", "")
                    new_out_nm = settings.get("output_name", "my_lora")
                    new_dim = settings.get("network_dim", 4)
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

                    # Sample image params
                    new_sample_enable = settings.get("sample_images", False)
                    new_sample_every_n = settings.get("sample_every_n_epochs", 1)
                    new_sample_prompt = settings.get("sample_prompt", "")
                    new_sample_negative = settings.get("sample_negative_prompt", "")
                    new_sample_w = settings.get("sample_w", new_w)
                    new_sample_h = settings.get("sample_h", new_h)

                    # Post-processing params
                    new_in_lora = settings.get("input_lora_path", "")
                    new_out_comfy = settings.get("output_comfy_lora_path", "")

                    # Load dataset config content
                    preview_content = load_dataset_config_content(path)

                    msg = f"Project initialized at {path}. "
                    if settings:
                        msg += " Settings loaded."
                    msg += " 'training' folder ready. Configure the dataset in the 'training' folder. Images and caption files (same name as image, extension is '.txt') should be placed in the 'training' folder."
                    msg += "\n\nプロジェクトが初期化されました。"
                    if settings:
                        msg += "設定が読み込まれました。"
                    msg += "'training' フォルダが準備されました。画像とキャプションファイル（画像と同じファイル名で拡張子は '.txt'）を配置してください。"

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
                        new_dim,
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
                        new_sample_enable,
                        new_sample_every_n,
                        new_sample_prompt,
                        new_sample_negative,
                        new_sample_w,
                        new_sample_h,
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
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            def generate_config(project_path, w, h, batch, model_val, vram_val, comfy_val, vae_val, te1_val, te2_val):
                if not project_path:
                    return "Error: Project directory not specified.\nエラー: プロジェクトディレクトリが指定されていません。", ""

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
                    return f"Successfully generated config at / 設定ファイルが作成されました: {config_path}", toml_content
                except Exception as e:
                    return f"Error generating config / 設定ファイルの生成に失敗しました: {str(e)}", ""

        with gr.Accordion(i18n("acc_preprocessing"), open=False):
            gr.Markdown(i18n("desc_preprocessing"))
            with gr.Row():
                set_preprocessing_defaults_btn = gr.Button(i18n("btn_set_paths"))
            with gr.Row():
                vae_path = gr.Textbox(label=i18n("lbl_vae_path"), placeholder=i18n("ph_vae_path"), max_lines=1)
                text_encoder1_path = gr.Textbox(label=i18n("lbl_te1_path"), placeholder=i18n("ph_te1_path"), max_lines=1)
                text_encoder2_path = gr.Textbox(label=i18n("lbl_te2_path"), placeholder=i18n("ph_te2_path"), max_lines=1)

            with gr.Row():
                cache_latents_btn = gr.Button(i18n("btn_cache_latents"))
                cache_text_btn = gr.Button(i18n("btn_cache_text"))

            # Simple output area for caching logs
            caching_output = gr.Textbox(label=i18n("lbl_cache_log"), lines=10, interactive=False)

            def validate_models_dir(path):
                if not path:
                    return "Please enter a ComfyUI models directory. / ComfyUIのmodelsディレクトリを入力してください。"

                required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                missing = []
                for d in required_subdirs:
                    if not os.path.exists(os.path.join(path, d)):
                        missing.append(d)

                if missing:
                    return f"Error: Missing subdirectories in models folder / modelsフォルダに以下のサブディレクトリが見つかりません: {', '.join(missing)}"

                return "Valid ComfyUI models directory structure found / 有効なComfyUI modelsディレクトリ構造が見つかりました。"

            def set_recommended_settings(project_path, model_arch, vram_val):
                w, h = config_manager.get_resolution(model_arch)
                recommended_batch_size = config_manager.get_batch_size(model_arch, vram_val)

                if project_path:
                    save_project_settings(project_path, resolution_w=w, resolution_h=h, batch_size=recommended_batch_size)
                return w, h, recommended_batch_size

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
                # Get number of images from project_path to adjust num_epochs later
                cache_dir = os.path.join(project_path, "cache")
                pattern = "*" + ("_qi" if model_arch == "Qwen-Image" else "_zi") + ".safetensors"
                num_images = len(glob.glob(os.path.join(cache_dir, pattern))) if os.path.exists(cache_dir) else 0

                # Get training defaults from config manager
                defaults = config_manager.get_training_defaults(model_arch, vram_val, comfy_models_dir)

                # Adjust num_epochs based on number of images (simple heuristic)
                default_num_steps = defaults.get("default_num_steps", 1000)
                if num_images > 0:
                    adjusted_epochs = max(1, int((default_num_steps / num_images)))
                else:
                    adjusted_epochs = 16  # Fallback default
                sample_every_n_epochs = (adjusted_epochs // 4) if adjusted_epochs >= 4 else 1

                dit_default = defaults.get("dit_path", "")
                dim = defaults.get("network_dim", 4)
                lr = defaults.get("learning_rate", 1e-4)
                epochs = adjusted_epochs
                save_n = defaults.get("save_every_n_epochs", 1)
                flow = defaults.get("discrete_flow_shift", 2.0)
                swap = defaults.get("block_swap", 0)
                prec = defaults.get("mixed_precision", "bf16")
                grad_cp = defaults.get("gradient_checkpointing", True)
                fp8_s = defaults.get("fp8_scaled", True)
                fp8_l = defaults.get("fp8_llm", True)

                sample_w_default, sample_h_default = config_manager.get_resolution(model_arch)

                if project_path:
                    save_project_settings(
                        project_path,
                        dit_path=dit_default,
                        network_dim=dim,
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
                        sample_every_n_epochs=sample_every_n_epochs,
                        sample_w=sample_w_default,
                        sample_h=sample_h_default,
                    )

                return (
                    dit_default,
                    dim,
                    lr,
                    epochs,
                    save_n,
                    flow,
                    swap,
                    prec,
                    grad_cp,
                    fp8_s,
                    fp8_l,
                    sample_every_n_epochs,
                    sample_w_default,
                    sample_h_default,
                )

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

                    output_log = command + "\n\n"
                    for line in process.stdout:
                        output_log += line
                        yield output_log

                    process.wait()
                    if process.returncode != 0:
                        output_log += (
                            f"\nError: Process exited with code / プロセスが次のコードでエラー終了しました: {process.returncode}"
                        )
                        yield output_log
                    else:
                        output_log += "\nProcess completed successfully / プロセスが正常に完了しました"
                        yield output_log

                except Exception as e:
                    yield f"Error executing command / コマンドの実行中にエラーが発生しました: {str(e)}"

            def cache_latents(project_path, vae_path_val, te1, te2, model, comfy, w, h, batch, vram_val):
                if not project_path:
                    yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
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
                    yield "Error: VAE path not set. / VAEのパスが設定されていません。"
                    return

                if not os.path.exists(vae_path_val):
                    yield f"Error: VAE model not found at / 指定されたパスにVAEモデルが見つかりません: {vae_path_val}"
                    return

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first. / dataset_config.tomlが {project_path} に見つかりません。先に設定ファイルを生成してください。"
                    return

                script_name = "zimage_cache_latents.py"
                if model == "Qwen-Image":
                    script_name = "qwen_image_cache_latents.py"

                script_path = os.path.join("src", "musubi_tuner", script_name)

                cmd = [sys.executable, script_path, "--dataset_config", config_path, "--vae", vae_path_val]

                # Placeholder for argument modification
                if model == "Z-Image-Turbo":
                    pass
                elif model == "Qwen-Image":
                    pass

                command_str = " ".join(cmd)
                yield f"Starting Latent Caching. Please wait for the first log to appear. / Latentのキャッシュを開始します。最初のログが表示されるまでにしばらくかかります。\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

            def cache_text_encoder(project_path, te1_path_val, te2_path_val, vae, model, comfy, w, h, batch, vram_val):
                if not project_path:
                    yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
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
                    yield "Error: Text Encoder 1 path not set. / Text Encoder 1のパスが設定されていません。"
                    return

                if not os.path.exists(te1_path_val):
                    yield f"Error: Text Encoder 1 model not found at / 指定されたパスにText Encoder 1モデルが見つかりません: {te1_path_val}"
                    return

                # Z-Image only uses te1 for now, but keeping te2 in signature if needed later or for other models

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first. / dataset_config.tomlが {project_path} に見つかりません。先に設定ファイルを生成してください。"
                    return

                script_name = "zimage_cache_text_encoder_outputs.py"
                if model == "Qwen-Image":
                    script_name = "qwen_image_cache_text_encoder_outputs.py"

                script_path = os.path.join("src", "musubi_tuner", script_name)

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

                # Model-specific argument modification
                if model == "Z-Image-Turbo":
                    pass
                elif model == "Qwen-Image":
                    # Add --fp8_vl for low VRAM (16GB or less)
                    if vram_val in ["12", "16"]:
                        cmd.append("--fp8_vl")

                command_str = " ".join(cmd)
                yield f"Starting Text Encoder Caching. Please wait for the first log to appear. / Text Encoderのキャッシュを開始します。最初のログが表示されるまでにしばらくかかります。\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

        with gr.Accordion(i18n("acc_training"), open=False):
            gr.Markdown(i18n("desc_training_basic"))
            training_model_info = gr.Markdown(i18n("desc_training_zimage"))

            with gr.Row():
                set_training_defaults_btn = gr.Button(i18n("btn_rec_params"))
            with gr.Row():
                dit_path = gr.Textbox(label=i18n("lbl_dit_path"), placeholder=i18n("ph_dit_path"), max_lines=1)

            with gr.Row():
                output_name = gr.Textbox(label=i18n("lbl_output_name"), value="my_lora", max_lines=1)

            with gr.Group():
                gr.Markdown(i18n("header_basic_params"))
                with gr.Row():
                    network_dim = gr.Number(label=i18n("lbl_dim"), value=4)
                    learning_rate = gr.Number(label=i18n("lbl_lr"), value=1e-4)
                    num_epochs = gr.Number(label=i18n("lbl_epochs"), value=16)
                    save_every_n_epochs = gr.Number(label=i18n("lbl_save_every"), value=1)

            with gr.Group():
                with gr.Row():
                    discrete_flow_shift = gr.Number(label=i18n("lbl_flow_shift"), value=2.0)
                    block_swap = gr.Slider(label=i18n("lbl_block_swap"), minimum=0, maximum=60, step=1, value=0)

                with gr.Accordion(i18n("accordion_advanced"), open=False):
                    gr.Markdown(i18n("desc_training_detailed"))

                with gr.Row():
                    mixed_precision = gr.Dropdown(label=i18n("lbl_mixed_precision"), choices=["bf16", "fp16", "no"], value="bf16")
                    gradient_checkpointing = gr.Checkbox(label=i18n("lbl_grad_cp"), value=True)

                with gr.Row():
                    fp8_scaled = gr.Checkbox(label=i18n("lbl_fp8_scaled"), value=True)
                    fp8_llm = gr.Checkbox(label=i18n("lbl_fp8_llm"), value=True)

            with gr.Group():
                gr.Markdown(i18n("header_sample_images"))
                sample_images = gr.Checkbox(label=i18n("lbl_enable_sample"), value=False)
                with gr.Row():
                    sample_prompt = gr.Textbox(label=i18n("lbl_sample_prompt"), placeholder=i18n("ph_sample_prompt"))
                with gr.Row():
                    sample_negative_prompt = gr.Textbox(
                        label=i18n("lbl_sample_negative_prompt"),
                        placeholder=i18n("ph_sample_negative_prompt"),
                    )
                with gr.Row():
                    sample_w = gr.Number(label=i18n("lbl_sample_w"), value=1024, precision=0)
                    sample_h = gr.Number(label=i18n("lbl_sample_h"), value=1024, precision=0)
                    sample_every_n = gr.Number(label=i18n("lbl_sample_every_n"), value=1, precision=0)

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
                input_lora = gr.Textbox(label=i18n("lbl_input_lora"), placeholder=i18n("ph_input_lora"), max_lines=1)
                output_comfy_lora = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder=i18n("ph_output_comfy"), max_lines=1)

            convert_btn = gr.Button(i18n("btn_convert"))
            conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)

        def convert_lora_to_comfy(project_path, input_path, output_path, model, comfy, w, h, batch, vae, te1, te2):
            if not project_path:
                yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
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
                yield "Error: Input and Output paths must be specified. / 入力・出力パスを指定してください。"
                return

            if not os.path.exists(input_path):
                yield f"Error: Input file not found at {input_path} / 入力ファイルが見つかりません: {input_path}"
                return

            # Script path
            script_path = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
            if not os.path.exists(script_path):
                yield f"Error: Conversion script not found at {script_path} / 変換スクリプトが見つかりません: {script_path}"
                return

            cmd = [sys.executable, script_path, input_path, output_path]

            command_str = " ".join(cmd)
            yield f"Starting Conversion. / 変換を開始します。\nCommand: {command_str}\n\n"

            yield from run_command(command_str)

        def start_training(
            project_path,
            model,
            dit,
            vae,
            te1,
            output_nm,
            dim,
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
            should_sample_images,
            sample_every_n,
            sample_prompt_val,
            sample_negative_prompt_val,
            sample_w_val,
            sample_h_val,
        ):
            import shlex

            if not project_path:
                return "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
            if not dit:
                return "Error: Base Model / DiT Path not set. / Base Model / DiTのパスが設定されていません。"
            if not os.path.exists(dit):
                return f"Error: Base Model / DiT file not found at {dit} / Base Model / DiTファイルが見つかりません: {dit}"
            if not vae:
                return "Error: VAE Path not set (configure in Preprocessing). / VAEのパスが設定されていません (Preprocessingで設定してください)。"
            if not te1:
                return "Error: Text Encoder 1 Path not set (configure in Preprocessing). / Text Encoder 1のパスが設定されていません (Preprocessingで設定してください)。"

            dataset_config = os.path.join(project_path, "dataset_config.toml")
            if not os.path.exists(dataset_config):
                return "Error: dataset_config.toml not found. Please generate it. / dataset_config.toml が見つかりません。生成してください。"

            output_dir = os.path.join(project_path, "models")
            logging_dir = os.path.join(project_path, "logs")

            # Save settings
            save_project_settings(
                project_path,
                dit_path=dit,
                output_name=output_nm,
                network_dim=dim,
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
                sample_images=should_sample_images,
                sample_every_n_epochs=sample_every_n,
                sample_prompt=sample_prompt_val,
                sample_negative_prompt=sample_negative_prompt_val,
                sample_w=sample_w_val,
                sample_h=sample_h_val,
            )

            # Model specific command modification
            if model == "Z-Image-Turbo":
                arch_name = "zimage"
            elif model == "Qwen-Image":
                arch_name = "qwen_image"

            # Construct command for cmd /c to run and then pause
            # We assume 'accelerate' is in the PATH.
            script_path = os.path.join("src", "musubi_tuner", f"{arch_name}_train_network.py")

            # Inner command list - arguments for accelerate launch
            inner_cmd = [
                "accelerate",
                "launch",
                # accelerate args: we don't configure default_config.yaml, so we need to specify all here
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                prec,
                "--dynamo_backend=no",
                "--gpu_ids",
                "all",
                "--machine_rank",
                "0",
                "--main_training_function",
                "main",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
                # script and its args
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
                f"networks.lora_{arch_name}",
                "--network_dim",
                str(int(dim)),
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
                "--logging_dir",
                logging_dir,
                "--log_with",
                "tensorboard",
            ]

            # Sample image generation options
            if should_sample_images:
                sample_prompt_path = os.path.join(project_path, "sample_prompt.txt")
                templates = {
                    # prompt, negative prompt, width, height, flow shift, steps, CFG scale, seed
                    "Qwen-Image": "{prompt} --n {neg} --w {w} --h {h} --fs 2.2 --s 20 --l 4.0 --d 1234",
                    "Z-Image-Turbo": "{prompt} --n {neg} --w {w} --h {h} --fs 3.0 --s 20 --l 5.0 --d 1234",
                }
                template = templates.get(model, templates["Z-Image-Turbo"])
                prompt_str = (sample_prompt_val or "").replace("\n", " ").strip()
                neg_str = (sample_negative_prompt_val or "").replace("\n", " ").strip()
                try:
                    w_int = int(sample_w_val)
                    h_int = int(sample_h_val)
                except Exception:
                    return "Error: Sample width/height must be integers. / サンプル画像の幅と高さは整数で指定してください。"

                line = template.format(prompt=prompt_str, neg=neg_str, w=w_int, h=h_int)
                try:
                    with open(sample_prompt_path, "w", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception as e:
                    return f"Error writing sample_prompt.txt / sample_prompt.txt の作成に失敗しました: {str(e)}"

                inner_cmd.extend(
                    [
                        "--sample_prompts",
                        sample_prompt_path,
                        "--sample_at_first",
                        "--sample_every_n_epochs",
                        str(int(sample_every_n)),
                    ]
                )

            if prec != "no":
                inner_cmd.extend(["--mixed_precision", prec])

            if grad_cp:
                inner_cmd.append("--gradient_checkpointing")

            if fp8_s:
                inner_cmd.append("--fp8_base")
                inner_cmd.append("--fp8_scaled")

            if fp8_l:
                if model == "Z-Image-Turbo":
                    inner_cmd.append("--fp8_llm")
                elif model == "Qwen-Image":
                    inner_cmd.append("--fp8_vl")

            if swap > 0:
                inner_cmd.extend(["--blocks_to_swap", str(int(swap))])

            inner_cmd.append("--sdpa")
            inner_cmd.append("--split_attn")

            # Model specific command modification
            if model == "Z-Image-Turbo":
                pass
            elif model == "Qwen-Image":
                pass

            # Parse and append additional args
            if add_args:
                try:
                    split_args = shlex.split(add_args)
                    inner_cmd.extend(split_args)
                except Exception as e:
                    return f"Error parsing additional arguments / 追加引数の解析に失敗しました: {str(e)}"

            # Construct the full command string for cmd /c
            # list2cmdline will quote arguments as needed for Windows
            inner_cmd_str = subprocess.list2cmdline(inner_cmd)

            # Chain commands: Run training -> echo message -> pause >nul (hides default message)
            final_cmd_str = f"{inner_cmd_str} & echo. & echo Training finished. Press any key to close this window... 学習が完了しました。このウィンドウを閉じるには任意のキーを押してください。 & pause >nul"

            try:
                # Open in new console window
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                # Pass explicit 'cmd', '/c', string to ensure proper execution
                subprocess.Popen(["cmd", "/c", final_cmd_str], creationflags=flags, shell=False)
                return f"Training started in a new window! / 新しいウィンドウで学習が開始されました！\nCommand: {inner_cmd_str}"
            except Exception as e:
                return f"Error starting training / 学習の開始に失敗しました: {str(e)}"

        def update_model_info(model):
            if model == "Z-Image-Turbo":
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
                network_dim,
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
                sample_images,
                sample_every_n,
                sample_prompt,
                sample_negative_prompt,
                sample_w,
                sample_h,
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
                network_dim,
                learning_rate,
                num_epochs,
                save_every_n_epochs,
                discrete_flow_shift,
                block_swap,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
                sample_every_n,
                sample_w,
                sample_h,
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
                vram_size,
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
                vram_size,
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
                network_dim,
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
                sample_images,
                sample_every_n,
                sample_prompt,
                sample_negative_prompt,
                sample_w,
                sample_h,
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
