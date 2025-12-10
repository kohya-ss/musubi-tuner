import gradio as gr
import os
import toml

def construct_ui():
    with gr.Blocks(title="Musubi Tuner GUI") as demo:
        gr.Markdown("# Musubi Tuner GUI")
        gr.Markdown("A simple frontend for training LoRA models with Musubi Tuner.")

        with gr.Accordion("1. Project Settings", open=True):
            with gr.Row():
                project_dir = gr.Textbox(label="Project Working Directory", placeholder="Absolute path to your project folder")
            
            # Placeholder for project initialization or loading
            init_btn = gr.Button("Initialize/Load Project")
            project_status = gr.Markdown("")

            # Old init_project removed, replaced by logic in 'Dataset Settings' block to handle loading settings.

        with gr.Accordion("2. Model & Dataset Configuration", open=True):
            with gr.Row():
                model_arch = gr.Dropdown(
                    label="Model Architecture",
                    choices=[
                        "Z-Image",
                    ],
                    value="Z-Image"
                )
                
            with gr.Row():
                comfy_models_dir = gr.Textbox(label="ComfyUI Models Directory", placeholder="Absolute path to ComfyUI/models")
            
            # Validation for ComfyUI models directory
            models_status = gr.Markdown("")
            validate_models_btn = gr.Button("Validate Models Directory")

            # Placeholder for Dataset Settings (Step 3)
            gr.Markdown("### 3. Dataset Settings")
            with gr.Row():
                resolution_w = gr.Number(label="Resolution Width", value=1024, precision=0)
                resolution_h = gr.Number(label="Resolution Height", value=1024, precision=0)
                batch_size = gr.Number(label="Batch Size", value=1, precision=0)

            gen_toml_btn = gr.Button("Generate Dataset Config")
            dataset_status = gr.Markdown("")
            toml_preview = gr.Code(label="TOML Preview", interactive=False)

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
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    )
                try:
                    os.makedirs(os.path.join(path, "training"), exist_ok=True)
                    
                    # Load settings if available
                    settings = load_project_settings(path)
                    new_model = settings.get("model_arch", "Z-Image")
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
                        new_model, new_comfy, new_w, new_h, new_batch,
                        preview_content,
                        new_vae, new_te1, new_te2,
                        new_dit, new_out_nm, new_lr, new_epochs, new_save_n,
                        new_flow, new_swap, new_prec, new_grad_cp, new_fp8_s, new_fp8_l,
                        new_in_lora, new_out_comfy
                    )
                except Exception as e:
                    return (
                        f"Error initializing project: {str(e)}", 
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update()
                    )



            def generate_config(project_path, w, h, batch, model_val, comfy_val, vae_val, te1_val, te2_val):
                if not project_path:
                    return "Error: Project directory not specified.", ""
                
                # Save project settings first
                save_project_settings(
                    project_path, 
                    model_arch=model_val, 
                    comfy_models_dir=comfy_val, 
                    resolution_w=w, 
                    resolution_h=h, 
                    batch_size=batch, 
                    vae_path=vae_val, 
                    text_encoder1_path=te1_val, 
                    text_encoder2_path=te2_val
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







        with gr.Accordion("4. Preprocessing", open=False):
            gr.Markdown("Pre-calculate latents and text encoder outputs to speed up training.")
            with gr.Row():
                vae_path = gr.Textbox(label="VAE Path", placeholder="Path to VAE model")
                text_encoder1_path = gr.Textbox(label="Text Encoder 1 Path", placeholder="Path to Text Encoder 1")
                text_encoder2_path = gr.Textbox(label="Text Encoder 2 Path", placeholder="Path to Text Encoder 2 (Optional)")
            
            with gr.Row():
                cache_latents_btn = gr.Button("Cache Latents")
                cache_text_btn = gr.Button("Cache Text Encoder Outputs")
            
            # Simple output area for caching logs
            caching_output = gr.Textbox(label="Caching Log Output", lines=10, interactive=False)

            def validate_models_dir(path):
                if not path:
                    return "Please enter a ComfyUI models directory.", "", "", ""
                
                required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                missing = []
                for d in required_subdirs:
                    if not os.path.join(path, d) or not os.path.exists(os.path.join(path, d)):
                        missing.append(d)
                
                if missing:
                    return f"Error: Missing subdirectories in models folder: {', '.join(missing)}", "", "", ""
                
                # Default paths for Z-Image
                vae_default = os.path.join(path, "vae", "ae.safetensors")
                te1_default = os.path.join(path, "text_encoders", "qwen_3_4b.safetensors")
                dit_default = os.path.join(path, "diffusion_models", "z_image_turbo_bf16.safetensors")
                
                return "Valid ComfyUI models directory structure found.", vae_default, te1_default, "", dit_default



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
                        encoding='utf-8',
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
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
                    text_encoder2_path=te2
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
                
                cmd = [
                    sys.executable,
                    script_path,
                    "--dataset_config", config_path,
                    "--vae", vae_path_val
                ]
                
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
                    text_encoder2_path=te2_path_val
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
                    "--dataset_config", config_path,
                    "--text_encoder", te1_path_val,
                    "--batch_size", "1" # Conservative default
                ]
                
                command_str = " ".join(cmd)
                yield f"Starting Text Encoder Caching...\nCommand: {command_str}\n\n"
                
                yield from run_command(command_str)



        with gr.Accordion("5. Training", open=False):
            with gr.Row():
                dit_path = gr.Textbox(label="Base Model / DiT Path", placeholder="Path to DiT model")
            
            with gr.Row():
                output_name = gr.Textbox(label="Output LoRA Name", value="my_lora")
                
            with gr.Group():
                gr.Markdown("### Basic Parameters")
                with gr.Row():
                    learning_rate = gr.Number(label="Learning Rate", value=1e-4)
                    num_epochs = gr.Number(label="Epochs", value=16)
                    save_every_n_epochs = gr.Number(label="Save Every N Epochs", value=1)
            
            with gr.Group():
                gr.Accordion("Advanced Parameters", open=False)
                with gr.Row():
                    discrete_flow_shift = gr.Number(label="Discrete Flow Shift", value=2.0)
                    block_swap = gr.Slider(label="Block Swap (0-28)", minimum=0, maximum=28, step=1, value=0)
                
                with gr.Row():
                    mixed_precision = gr.Dropdown(label="Mixed Precision", choices=["bf16", "fp16", "no"], value="bf16")
                    gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True)
                
                with gr.Row():
                    fp8_scaled = gr.Checkbox(label="FP8 Scaled (DiT) - Enables --fp8_base and --fp8_scaled", value=True)
                    fp8_llm = gr.Checkbox(label="FP8 LLM (Text Encoder)", value=True)
            
            training_status = gr.Markdown("")
            start_training_btn = gr.Button("Start Training (New Window)", variant="primary")
            
        with gr.Accordion("6. Post-Processing", open=False):
            gr.Markdown("Convert Z-Image LoRA to ComfyUI format.")
            with gr.Row():
                input_lora = gr.Textbox(label="Input LoRA Path", placeholder="Path to trained .safetensors file")
                output_comfy_lora = gr.Textbox(label="Output ComfyUI LoRA Path", placeholder="Path to save converted model")
            
            convert_btn = gr.Button("Convert to ComfyUI Format")
            conversion_log = gr.Textbox(label="Conversion Log", lines=5, interactive=False)

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
                output_comfy_lora_path=output_path
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

            cmd = [
                sys.executable,
                script_path,
                input_path,
                output_path
            ]
            
            command_str = " ".join(cmd)
            yield f"Starting Conversion...\nCommand: {command_str}\n\n"
            
            yield from run_command(command_str)

        def start_training(project_path, dit, vae, te1, output_nm, lr, epochs, save_n, flow_shift, swap, prec, grad_cp, fp8_s, fp8_l):
            if not project_path:
                return "Error: Project directory not set.", "", ""
            if not dit:
                return "Error: Base Model / DiT Path not set.", "", ""
            if not vae:
                return "Error: VAE Path not set (configure in Preprocessing).", "", ""
            if not te1:
                return "Error: Text Encoder 1 Path not set (configure in Preprocessing).", "", ""

            dataset_config = os.path.join(project_path, "dataset_config.toml")
            if not os.path.exists(dataset_config):
                return "Error: dataset_config.toml not found. Please generate it.", "", ""
            
            output_dir = os.path.join(project_path, "models")
            
            # Predict output paths for post-processing
            predicted_input_lora = os.path.join(output_dir, f"{output_nm}.safetensors")
            predicted_output_comfy = os.path.join(output_dir, f"{output_nm}_comfy.safetensors")
            
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
                input_lora_path=predicted_input_lora,
                output_comfy_lora_path=predicted_output_comfy
            )
            
            # Construct command for cmd /c to run and then pause
            # We assume 'accelerate' is in the PATH.
            script_path = os.path.join("src", "musubi_tuner", "zimage_train_network.py")
            
            # Inner command list - arguments for accelerate launch
            inner_cmd = [
                "accelerate", "launch",
                "--num_cpu_threads_per_process", "1",
                "--mixed_precision", prec,
                script_path,
                "--dit", dit,
                "--vae", vae,
                "--text_encoder", te1,
                "--dataset_config", dataset_config,
                "--output_dir", output_dir,
                "--output_name", output_nm,
                "--network_module", "networks.lora_zimage", 
                "--network_dim", "32",
                "--optimizer_type", "adamw8bit",
                "--learning_rate", str(lr),
                "--max_train_epochs", str(int(epochs)),
                "--save_every_n_epochs", str(int(save_n)),
                "--timestep_sampling", "shift",
                "--weighting_scheme", "none",
                "--discrete_flow_shift", str(flow_shift),
                "--max_data_loader_n_workers", "2",
                "--persistent_data_loader_workers",
                "--seed", "42"
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

            # Construct the full command string for cmd /c
            # list2cmdline will quote arguments as needed for Windows
            inner_cmd_str = subprocess.list2cmdline(inner_cmd)
            
            # Chain commands: Run training -> echo message -> pause >nul (hides default message)
            final_cmd_str = f'{inner_cmd_str} & echo. & echo Training finished. Press any key to close this window... & pause >nul'

            try:
                # Open in new console window
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                # Pass explicit 'cmd', '/c', string to ensure proper execution
                subprocess.Popen(['cmd', '/c', final_cmd_str], creationflags=flags, shell=False)
                return f"Training started in a new window!\n\nCommand: {inner_cmd_str}", predicted_input_lora, predicted_output_comfy
            except Exception as e:
                return f"Error starting training: {str(e)}", predicted_input_lora, predicted_output_comfy

        # Event wiring moved to end to prevent UnboundLocalError
        init_btn.click(
            fn=init_project, 
            inputs=[project_dir], 
            outputs=[
                project_status, 
                model_arch, 
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
                input_lora,
                output_comfy_lora
            ]
        )
        
        gen_toml_btn.click(
            fn=generate_config, 
            inputs=[
                project_dir, resolution_w, resolution_h, batch_size, model_arch, comfy_models_dir,
                vae_path, text_encoder1_path, text_encoder2_path
            ], 
            outputs=[dataset_status, toml_preview]
        )

        validate_models_btn.click(
              fn=validate_models_dir,
              inputs=[comfy_models_dir],
              outputs=[models_status, vae_path, text_encoder1_path, text_encoder2_path, dit_path]
         )

        cache_latents_btn.click(
            fn=cache_latents,
            inputs=[
                project_dir, vae_path, text_encoder1_path, text_encoder2_path,
                model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size
            ],
            outputs=[caching_output]
        )

        cache_text_btn.click(
            fn=cache_text_encoder,
            inputs=[
                project_dir, text_encoder1_path, text_encoder2_path, vae_path,
                model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size
            ],
            outputs=[caching_output]
        )

        start_training_btn.click(
            fn=start_training,
            inputs=[
                project_dir, dit_path, vae_path, text_encoder1_path, output_name,
                learning_rate, num_epochs, save_every_n_epochs, discrete_flow_shift, block_swap,
                mixed_precision, gradient_checkpointing, fp8_scaled, fp8_llm
            ],
            outputs=[training_status, input_lora, output_comfy_lora]
        )

        convert_btn.click(
            fn=convert_lora_to_comfy,
            inputs=[
                project_dir, input_lora, output_comfy_lora, 
                model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size, 
                vae_path, text_encoder1_path, text_encoder2_path
            ],
            outputs=[conversion_log]
        )

    return demo

if __name__ == "__main__":
    demo = construct_ui()
    demo.launch()
