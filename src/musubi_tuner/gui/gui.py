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

            def save_project_settings(project_path, model, comfy_dir, w, h, batch):
                try:
                    settings = {
                        "model_arch": model,
                        "comfy_models_dir": comfy_dir,
                        "resolution_w": w,
                        "resolution_h": h,
                        "batch_size": batch,
                    }
                    settings_path = os.path.join(project_path, "musubi_project.toml")
                    with open(settings_path, "w", encoding="utf-8") as f:
                        toml.dump(settings, f)
                except Exception as e:
                    print(f"Error saving project settings: {e}")

            def init_project(path):
                if not path:
                    return (
                        "Please enter a project directory path.", 
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    )
                try:
                    os.makedirs(os.path.join(path, "training"), exist_ok=True)
                    
                    # Load settings if available
                    settings = load_project_settings(path)
                    new_model = settings.get("model_arch", "Z-Image") # Default fallback
                    new_comfy = settings.get("comfy_models_dir", "")
                    new_w = settings.get("resolution_w", 1024)
                    new_h = settings.get("resolution_h", 1024)
                    new_batch = settings.get("batch_size", 1)
                    
                    # Load dataset config content
                    preview_content = load_dataset_config_content(path)

                    msg = f"Project initialized at {path}. 'training' folder ready."
                    if settings:
                         msg += " Settings loaded."

                    return (
                        msg,
                        new_model,
                        new_comfy,
                        new_w,
                        new_h,
                        new_batch,
                        preview_content
                    )
                except Exception as e:
                    return (
                        f"Error initializing project: {str(e)}", 
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                    )

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
                    toml_preview
                ]
            )

            def generate_config(project_path, w, h, batch, model_val, comfy_val):
                if not project_path:
                    return "Error: Project directory not specified.", ""
                
                # Save project settings first
                save_project_settings(project_path, model_val, comfy_val, w, h, batch)

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

            gen_toml_btn.click(
                fn=generate_config, 
                inputs=[project_dir, resolution_w, resolution_h, batch_size, model_arch, comfy_models_dir], 
                outputs=[dataset_status, toml_preview]
            )

            def validate_models_dir(path):
                if not path:
                    return "Please enter the ComfyUI models directory."
                
                required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                missing = []
                for d in required_subdirs:
                    if not os.path.isdir(os.path.join(path, d)):
                        missing.append(d)
                
                if missing:
                    return f"Error: Missing subdirectories in models folder: {', '.join(missing)}"
                
                return "Valid ComfyUI models directory structure found."

            validate_models_btn.click(fn=validate_models_dir, inputs=[comfy_models_dir], outputs=[models_status])

        with gr.Accordion("3. Data Preparation (Caching)", open=False):
            gr.Markdown("Pre-calculate latents and text encoder outputs to speed up training.")
            with gr.Row():
                vae_path = gr.Textbox(label="VAE Path")
                text_encoder1_path = gr.Textbox(label="Text Encoder 1 Path")
                text_encoder2_path = gr.Textbox(label="Text Encoder 2 Path")
            
            with gr.Row():
                cache_latents_btn = gr.Button("Cache Latents")
                cache_text_btn = gr.Button("Cache Text Encoder Outputs")
            
            # Simple output area for caching logs
            caching_output = gr.Textbox(label="Caching Log Output", lines=10, interactive=False)

        with gr.Accordion("4. Training", open=False):
            with gr.Row():
                base_model_path = gr.Textbox(label="Base Model / DiT Path")
            
            with gr.Group():
                gr.Markdown("### Basic Parameters")
                with gr.Row():
                    learning_rate = gr.Number(label="Learning Rate", value=1e-4)
                    num_epochs = gr.Number(label="Epochs", value=10)
                    batch_size = gr.Number(label="Batch Size", value=1)
            
            with gr.Group():
                gr.Accordion("Advanced Parameters", open=False)
                # Add advanced params later
            
            start_training_btn = gr.Button("Start Training", variant="primary")
            stop_training_btn = gr.Button("Stop Training", variant="stop")
            
            training_output = gr.Textbox(label="Training Log Output", lines=20, interactive=False, autoscroll=True)

        with gr.Accordion("5. Post-Processing", open=False):
            gr.Markdown("Convert or Merge LoRA models.")
            with gr.Tab("Convert LoRA"):
                input_lora = gr.Textbox(label="Input LoRA Path")
                target_format = gr.Dropdown(label="Target Format", choices=["diffusers", "comfy", "other"])
                convert_btn = gr.Button("Convert")
            
            with gr.Tab("Merge LoRA"):
                # Placeholder for merge UI
                pass

        # Event wiring can happen later
        
    return demo

if __name__ == "__main__":
    demo = construct_ui()
    demo.launch()
