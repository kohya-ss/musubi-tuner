import gradio as gr
import os

def construct_ui():
    with gr.Blocks(title="Musubi Tuner GUI") as demo:
        gr.Markdown("# Musubi Tuner GUI")
        gr.Markdown("A simple frontend for training LoRA models with Musubi Tuner.")

        with gr.Accordion("1. Project Settings", open=True):
            with gr.Row():
                project_dir = gr.Textbox(label="Project Working Directory", placeholder="Absolute path to your project folder")
                output_dir = gr.Textbox(label="Output Directory", placeholder="Where to save trained models")
            
            # Placeholder for project initialization or loading
            init_btn = gr.Button("Initialize/Load Project")

        with gr.Accordion("2. Model & Dataset Configuration", open=True):
            with gr.Row():
                model_arch = gr.Dropdown(
                    label="Model Architecture",
                    choices=[
                        "HunyuanVideo",
                        "HunyuanVideo 1.5",
                        "Wan2.1",
                        "FLUX.1 Kontext",
                        "Z-Image",
                        "Qwen-Image",
                        "FramePack"
                    ],
                    value="HunyuanVideo"
                )
                
            with gr.Row():
                train_data_dir = gr.Textbox(label="Training Images/Videos Directory")
                # In a real app, we might want a file picker here, but text is fine for now
            
            dataset_config_path = gr.Textbox(label="Dataset Config Path (TOML)", placeholder="Auto-generated or path to existing TOML")
            
            gen_toml_btn = gr.Button("Generate/Update Dataset Config")
            toml_preview = gr.Code(label="TOML Preview", language="toml", interactive=True)

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
