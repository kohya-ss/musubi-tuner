import glob
import gradio as gr
import os
import functools
import json
import toml
import sys
import subprocess
import shlex
import locale
from musubi_tuner.gui.config_manager import ConfigManager
from musubi_tuner.gui.dataset_utils import DatasetStateManager
from musubi_tuner.gui import arg_loader

config_manager = ConfigManager()

def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

I18N_CACHE = None
CURRENT_LANG = None

def load_i18n_data():
    global I18N_CACHE
    if I18N_CACHE is not None:
        return I18N_CACHE
        
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_data.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            I18N_CACHE = json.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load i18n_data.json from {json_path}: {e}")
        I18N_CACHE = {"en": {}}
    return I18N_CACHE

def get_system_lang():
    global CURRENT_LANG
    if CURRENT_LANG:
        return CURRENT_LANG
    
    lang = "en"
    try:
        env_lang = os.environ.get("LANG", "").lower()
        if env_lang.startswith("ja"):
            lang = "ja"
        else:
            sys_lang = locale.getdefaultlocale()[0]
            if sys_lang and sys_lang.lower().startswith("ja"):
                lang = "ja"
    except Exception:
        pass
    
    CURRENT_LANG = lang
    return lang

def i18n(key):
    lang = get_system_lang()
    data = load_i18n_data()
    return data.get(lang, {}).get(key, data.get("en", {}).get(key, key))

def construct_ui():
    custom_css = load_custom_css()
    
    with gr.Group(elem_classes=["main-container"]) as demo:
        gr.Markdown(i18n("app_header"))
        gr.Markdown(i18n("app_desc"))

        with gr.Tabs() as tabs:
            with gr.TabItem(i18n("tab_init"), id="init"):
                gr.Markdown(i18n("desc_project"))
                with gr.Row():
                    project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"), max_lines=1)
                
                init_btn = gr.Button(i18n("btn_init_project"), variant="primary")
                project_status = gr.Markdown("")

            with gr.TabItem(i18n("tab_model"), id="model"):
                with gr.Row():
                    model_arch = gr.Dropdown(
                        label=i18n("lbl_model_arch"),
                        choices=[
                            "Wan 2.1 (T2V-14B)", "Wan 2.1 (I2V-14B)", 
                            "Wan 2.2 (T2V-5B)", "Wan 2.2 (T2V-14B)", "Wan 2.2 (I2V-14B)",
                            "HunyuanVideo", "Hunyuan 1.5", 
                            "FramePack", "Flux.1 Kontext", "Kandinsky 5", "Qwen-Image", "Z-Image-Turbo",
                        ],
                        value="Wan 2.2 (T2V-14B)",
                        interactive=True,
                    )
                    vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24", interactive=True)

                with gr.Group():
                    gr.Markdown("### ComfyUI & Paths")
                    with gr.Row():
                        comfy_models_dir = gr.Textbox(label=i18n("lbl_comfy_dir"), placeholder=i18n("ph_comfy_dir"), max_lines=1, interactive=True)
                        validate_models_btn = gr.Button(i18n("btn_validate_models"), scale=0)
                    models_status = gr.Markdown("")

                    with gr.Row():
                        dit_path = gr.Textbox(label=i18n("lbl_dit"), max_lines=1, interactive=True)
                        vae_path = gr.Textbox(label=i18n("lbl_vae"), max_lines=1, interactive=True)
                    with gr.Row():
                        text_encoder1_path = gr.Textbox(label=i18n("lbl_te1"), max_lines=1, interactive=True)
                        text_encoder2_path = gr.Textbox(label=i18n("lbl_te2"), max_lines=1, interactive=True)

                    with gr.Row(visible=False) as row_wan_paths:
                        wan_t5 = gr.Textbox(label=i18n("lbl_wan_t5"), max_lines=1, interactive=True)
                        wan_clip = gr.Textbox(label=i18n("lbl_wan_clip"), max_lines=1, interactive=True)
                    with gr.Row(visible=False) as row_wan_2_2_paths:
                        wan_dit_high_noise = gr.Textbox(label=i18n("lbl_wan_dit_high_noise"), max_lines=1, interactive=True)
                        wan_timestep_boundary = gr.Number(label=i18n("lbl_wan_timestep_boundary"), value=None, interactive=True)
                    
                    with gr.Row(visible=False) as row_extra_models:
                        img_enc_path = gr.Textbox(label=i18n("lbl_img_enc"), max_lines=1)
                        byt5_path = gr.Textbox(label=i18n("lbl_byt5"), max_lines=1)
                    
                    with gr.Row():
                        set_paths_btn = gr.Button(i18n("btn_set_paths"))
                        rec_params_btn = gr.Button(i18n("btn_rec_params"))

                with gr.Accordion("Performance & Memory", open=False):
                    with gr.Row():
                        mixed_precision = gr.Dropdown(label=i18n("lbl_mixed_precision"), choices=["bf16", "fp16", "no"], value="bf16")
                        gradient_checkpointing = gr.Checkbox(label=i18n("lbl_gradient_checkpointing"), value=True)
                        compile_flag = gr.Checkbox(label=i18n("lbl_compile"), value=False)
                    
                    with gr.Row():
                        fp8_scaled = gr.Checkbox(label=i18n("lbl_fp8_scaled"), value=True)
                        fp8_llm = gr.Checkbox(label=i18n("lbl_fp8_llm"), value=True)
                        offload_txt_in = gr.Checkbox(label=i18n("lbl_offload_txt_in"), value=False)

                    with gr.Row():
                        blocks_to_swap = gr.Slider(label=i18n("lbl_blocks_to_swap"), minimum=0, maximum=60, step=1, value=0)
                        use_pinned_memory_for_block_swap = gr.Checkbox(label=i18n("lbl_use_pinned_memory_for_block_swap"), value=False)

                    with gr.Row():
                        attn_mode = gr.Dropdown(label=i18n("lbl_attn_mode"), choices=["sdpa", "flash_attn", "sage_attn", "xformers", "split_attn"], value="sdpa", interactive=True)

                with gr.Accordion("Advanced Model Settings", open=False):
                    with gr.Group(visible=False) as arg_group_wan:
                        with gr.Row():
                            wan_fp8_scaled = gr.Checkbox(label=i18n("lbl_wan_fp8_scaled"), value=True, interactive=True)
                            wan_fp8_t5 = gr.Checkbox(label=i18n("lbl_wan_fp8_t5"), value=True, interactive=True)
                            wan_vae_cache_cpu = gr.Checkbox(label=i18n("lbl_wan_vae_cache_cpu"), value=False, interactive=True)
                            wan_one_frame = gr.Checkbox(label=i18n("lbl_wan_one_frame"), value=False, interactive=True)
                        with gr.Row():
                            wan_offload_inactive_dit = gr.Checkbox(label=i18n("lbl_wan_offload_inactive_dit"), value=False, interactive=True)
                            wan_force_v2_1_time_embedding = gr.Checkbox(label=i18n("lbl_wan_force_v2_1_time_embedding"), value=False, interactive=True)

                    with gr.Group(visible=False) as arg_group_wan_2_2:
                        gr.Markdown("*Wan 2.2 specific settings")

                    with gr.Group(visible=False) as arg_group_hv:
                        with gr.Row():
                            hv_dit = gr.Textbox(label=i18n("lbl_hv_dit"), max_lines=1, interactive=True)
                            hv_vae = gr.Textbox(label=i18n("lbl_hv_vae"), max_lines=1, interactive=True)
                        with gr.Row():
                            hv_te1 = gr.Textbox(label=i18n("lbl_hv_te1"), max_lines=1, interactive=True)
                            hv_te2 = gr.Textbox(label=i18n("lbl_hv_te2"), max_lines=1, interactive=True)
                        with gr.Row():
                            hv_fp8_llm = gr.Checkbox(label=i18n("lbl_hv_fp8_llm"), value=True, interactive=True)
                            hv_fp8_scaled = gr.Checkbox(label=i18n("lbl_hv_fp8_scaled"), value=True, interactive=True)
                            hv_vae_tiling = gr.Checkbox(label=i18n("lbl_hv_vae_tiling"), value=False, interactive=True)
                            hv_vae_chunk_size = gr.Number(label=i18n("lbl_hv_vae_chunk_size"), value=None, precision=0, interactive=True)

                    with gr.Group(visible=False) as arg_group_hv15:
                        with gr.Row():
                            hv15_task = gr.Dropdown(label=i18n("lbl_hv15_task"), choices=["t2v", "i2v"], value="t2v", interactive=True)
                            hv15_dit = gr.Textbox(label=i18n("lbl_hv15_dit"), max_lines=1, interactive=True)
                        with gr.Row():
                            hv15_master_te = gr.Textbox(label=i18n("lbl_hv15_master_te"), max_lines=1, interactive=True)
                            hv15_byt5 = gr.Textbox(label=i18n("lbl_hv15_byt5"), max_lines=1, interactive=True)
                            hv15_img_enc = gr.Textbox(label=i18n("lbl_hv15_img_enc"), max_lines=1, interactive=True)
                        with gr.Row():
                            hv15_fp8_vl = gr.Checkbox(label=i18n("lbl_hv15_fp8_vl"), value=True, interactive=True)
                            hv15_fp8_scaled = gr.Checkbox(label=i18n("lbl_hv15_fp8_scaled"), value=True, interactive=True)
                            hv15_vae_sample_size = gr.Number(label=i18n("lbl_hv15_vae_sample_size"), value=None, precision=0, interactive=True)

                    with gr.Group(visible=False) as arg_group_fpack:
                        with gr.Row():
                            fpack_te1 = gr.Textbox(label=i18n("lbl_fpack_te1"), max_lines=1, interactive=True)
                            fpack_te2 = gr.Textbox(label=i18n("lbl_fpack_te2"), max_lines=1, interactive=True)
                            fpack_img_enc = gr.Textbox(label=i18n("lbl_fpack_img_enc"), max_lines=1, interactive=True)
                        with gr.Row():
                            fpack_latent_window_size = gr.Number(label=i18n("lbl_fpack_latent_window_size"), value=9, precision=0, interactive=True)
                            fpack_bulk_decode = gr.Checkbox(label=i18n("lbl_fpack_bulk_decode"), value=False, interactive=True)
                            fpack_f1 = gr.Checkbox(label=i18n("lbl_fpack_f1"), value=False, interactive=True)
                            fpack_one_frame = gr.Checkbox(label=i18n("lbl_fpack_one_frame"), value=False, interactive=True)

                    with gr.Group(visible=False) as arg_group_flux:
                        with gr.Row():
                            flux_te1 = gr.Textbox(label=i18n("lbl_flux_te1"), max_lines=1, interactive=True)
                            flux_te2 = gr.Textbox(label=i18n("lbl_flux_te2"), max_lines=1, interactive=True)
                        with gr.Row():
                            flux_fp8_t5 = gr.Checkbox(label=i18n("lbl_flux_fp8_t5"), value=True, interactive=True)
                            flux_fp8_scaled = gr.Checkbox(label=i18n("lbl_flux_fp8_scaled"), value=True, interactive=True)

                    with gr.Group(visible=False) as arg_group_k5:
                        with gr.Row():
                            k5_task = gr.Textbox(label=i18n("lbl_k5_task"), placeholder="e.g. pro-t2v-1024", max_lines=1, interactive=True)
                            k5_scheduler_scale = gr.Number(label=i18n("lbl_k5_scheduler_scale"), value=1.0, interactive=True)
                            k5_force_nabla_attention = gr.Checkbox(label=i18n("lbl_k5_force_nabla_attention"), value=False, interactive=True)

                    with gr.Group(visible=False) as arg_group_qwen:
                        with gr.Row():
                            qwen_te = gr.Textbox(label=i18n("lbl_qwen_te"), max_lines=1, interactive=True)
                            qwen_fp8_vl = gr.Checkbox(label=i18n("lbl_qwen_fp8_vl"), value=True, interactive=True)
                            qwen_fp8_scaled = gr.Checkbox(label=i18n("lbl_qwen_fp8_scaled"), value=True, interactive=True)

                    with gr.Group(visible=False) as arg_group_zimage:
                        with gr.Row():
                            zimage_te = gr.Textbox(label=i18n("lbl_zimage_te"), max_lines=1, interactive=True)
                            zimage_fp8_llm = gr.Checkbox(label=i18n("lbl_zimage_fp8_llm"), value=True, interactive=True)
                            zimage_fp8_scaled = gr.Checkbox(label=i18n("lbl_zimage_fp8_scaled"), value=True, interactive=True)
                            zimage_use_32bit_attention = gr.Checkbox(label=i18n("lbl_zimage_use_32bit_attention"), value=False, interactive=True)

                def update_model_ui(arch):
                    is_wan = "Wan" in arch
                    is_wan_2_1 = "Wan 2.1" in arch
                    is_wan_2_2 = "Wan 2.2" in arch
                    is_hv = arch == "HunyuanVideo"
                    is_hv15 = arch == "Hunyuan 1.5"
                    is_fpack = arch == "FramePack"
                    is_flux = "Flux" in arch
                    is_k5 = arch == "Kandinsky 5"
                    is_qwen = "Qwen" in arch
                    is_zimage = "Z-Image" in arch

                    return [
                        gr.update(visible=is_wan),
                        gr.update(visible=is_wan_2_2),
                        gr.update(visible=is_wan or is_hv15 or is_fpack),
                        gr.update(visible=is_wan),
                        gr.update(visible=is_wan_2_2),
                        gr.update(visible=is_hv),
                        gr.update(visible=is_hv15),
                        gr.update(visible=is_fpack),
                        gr.update(visible=is_flux),
                        gr.update(visible=is_k5),
                        gr.update(visible=is_qwen),
                        gr.update(visible=is_zimage),
                    ]

                model_arch.change(
                    fn=update_model_ui,
                    inputs=[model_arch],
                    outputs=[
                        row_wan_paths, row_wan_2_2_paths, row_extra_models,
                        arg_group_wan, arg_group_wan_2_2, 
                        arg_group_hv, arg_group_hv15, arg_group_fpack, 
                        arg_group_flux, arg_group_k5, arg_group_qwen, arg_group_zimage
                    ]
                )

            with gr.TabItem(i18n("tab_dataset"), id="dataset"):
                gr.Markdown(i18n("desc_dataset"))
                dataset_state = gr.State([])
                with gr.Row():
                    resolution_w = gr.Number(label=i18n("lbl_res_w"), value=1024, precision=0, interactive=True)
                    resolution_h = gr.Number(label=i18n("lbl_res_h"), value=1024, precision=0, interactive=True)
                    num_repeats = gr.Number(label=i18n("lbl_num_repeats"), value=1, precision=0, interactive=True)
                    batch_size = gr.Number(label=i18n("lbl_batch_size"), value=1, precision=0, interactive=True)
                    set_rec_settings_btn = gr.Button(i18n("btn_rec_res_batch"))
                with gr.Row():
                    caption_extension = gr.Textbox(label=i18n("lbl_caption_extension"), value=".txt", interactive=True)
                    enable_bucket = gr.Checkbox(label=i18n("lbl_enable_bucket"), value=False, interactive=True)
                    bucket_no_upscale = gr.Checkbox(label=i18n("lbl_bucket_no_upscale"), value=False, interactive=True)
                gr.Markdown("---")
                with gr.Row():
                    add_img_ds_btn = gr.Button(i18n("lbl_add_img_ds"))
                    add_vid_ds_btn = gr.Button(i18n("lbl_add_vid_ds"))

                @gr.render(inputs=[dataset_state])
                def render_datasets(state):
                     from dataclasses import asdict
                     if not state: 
                         gr.Markdown("📁 No datasets added yet.")
                         return
                     for i, ds in enumerate(state):
                         if hasattr(ds, "__dataclass_fields__"): data = asdict(ds)
                         else: data = ds
                         is_video = "video_directory" in data
                         type_str = "🎬 Video" if is_video else "🖼️ Image"
                         
                         with gr.Group(elem_classes=["dataset-card"]):
                             gr.Markdown(f"### {type_str} Dataset {i+1}", elem_classes=["dataset-card-header"])
                             with gr.Row():
                                 if not is_video:
                                     img_dir = gr.Textbox(label=f"📂 {i18n('lbl_image_directory')}", value=data.get("image_directory", ""), interactive=True, scale=2)
                                     img_dir.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="image_directory"), inputs=[dataset_state, img_dir], outputs=[dataset_state])
                                 else:
                                     vid_dir = gr.Textbox(label=f"📂 {i18n('lbl_video_directory')}", value=data.get("video_directory", ""), interactive=True, scale=2)
                                     vid_dir.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="video_directory"), inputs=[dataset_state, vid_dir], outputs=[dataset_state])
                                 cache_dir_ds = gr.Textbox(label=f"📦 {i18n('lbl_cache_directory')}", value=data.get("cache_directory", ""), interactive=True, scale=1)
                                 cache_dir_ds.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="cache_directory"), inputs=[dataset_state, cache_dir_ds], outputs=[dataset_state])
                             
                             with gr.Row():
                                 num_rep = gr.Number(label=f"🔁 {i18n('lbl_num_repeats')}", value=data.get("num_repeats", 1), precision=0, interactive=True)
                                 num_rep.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="num_repeats"), inputs=[dataset_state, num_rep], outputs=[dataset_state])
                                 batch_sz = gr.Number(label=f"📊 {i18n('lbl_batch_size')}", value=data.get("batch_size", 1), precision=0, interactive=True)
                                 batch_sz.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="batch_size"), inputs=[dataset_state, batch_sz], outputs=[dataset_state])
                                 cap_ext_ds = gr.Textbox(label=f"📝 {i18n('lbl_caption_extension')}", value=data.get("caption_extension", ".txt"), interactive=True)
                                 cap_ext_ds.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="caption_extension"), inputs=[dataset_state, cap_ext_ds], outputs=[dataset_state])
                                 res = gr.Textbox(label=f"📐 {i18n('lbl_resolution')}", value=f"{data.get('resolution', [1024, 1024])[0]}, {data.get('resolution', [1024, 1024])[1]}", interactive=True)
                                 res.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="resolution"), inputs=[dataset_state, res], outputs=[dataset_state])
                             
                             with gr.Accordion(f"⚙️ {i18n('dataset_adv_opts')}", open=False):
                                 with gr.Row():
                                     jsonl_label = i18n("lbl_video_jsonl_file") if is_video else i18n("lbl_image_jsonl_file")
                                     jsonl = gr.Textbox(label=jsonl_label, value=data.get("video_jsonl_file" if is_video else "image_jsonl_file", ""), interactive=True)
                                     jsonl.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="jsonl_file"), inputs=[dataset_state, jsonl], outputs=[dataset_state])
                                     ctrl_dir = gr.Textbox(label=i18n("lbl_control_directory"), value=data.get("control_directory", ""), interactive=True)
                                     ctrl_dir.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="control_directory"), inputs=[dataset_state, ctrl_dir], outputs=[dataset_state])
                                 if is_video:
                                     with gr.Row():
                                         target_f = gr.Textbox(label=i18n("lbl_target_frames"), value=", ".join(map(str, data.get("target_frames", [1]))), interactive=True)
                                         target_f.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="target_frames"), inputs=[dataset_state, target_f], outputs=[dataset_state])
                                         extract = gr.Dropdown(label=i18n("lbl_frame_extraction"), choices=["head", "chunk", "slide", "uniform", "full"], value=data.get("frame_extraction", "head"), interactive=True)
                                         extract.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="frame_extraction"), inputs=[dataset_state, extract], outputs=[dataset_state])
                                     with gr.Row():
                                         stride = gr.Number(label=i18n("lbl_frame_stride"), value=data.get("frame_stride", 1), precision=0, interactive=True)
                                         stride.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="frame_stride"), inputs=[dataset_state, stride], outputs=[dataset_state])
                                         sample = gr.Number(label=i18n("lbl_frame_sample"), value=data.get("frame_sample", 1), precision=0, interactive=True)
                                         sample.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="frame_sample"), inputs=[dataset_state, sample], outputs=[dataset_state])
                                         max_f = gr.Number(label=i18n("lbl_max_frames"), value=data.get("max_frames", 129), precision=0, interactive=True)
                                         max_f.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="max_frames"), inputs=[dataset_state, max_f], outputs=[dataset_state])
                                         src_fps = gr.Number(label=i18n("lbl_source_fps"), value=data.get("source_fps", None), interactive=True)
                                         src_fps.change(fn=functools.partial(save_dataset_field_dynamic, index=i, field="source_fps"), inputs=[dataset_state, src_fps], outputs=[dataset_state])
                             
                             with gr.Row(elem_classes=["dataset-card-footer"]):
                                 rm_btn = gr.Button(f"🗑️ {i18n('btn_remove_dataset')}", variant="stop")
                                 rm_btn.click(fn=functools.partial(remove_dataset_dynamic, index=i), inputs=[dataset_state], outputs=[dataset_state])

                gen_toml_btn = gr.Button(i18n("btn_gen_config"), variant="primary")
                dataset_status = gr.Markdown("")
                toml_preview = gr.Code(label=i18n("lbl_toml_preview"), interactive=False, visible=False)

            with gr.TabItem(i18n("tab_training"), id="training"):
                with gr.Row():
                    cache_latents_btn = gr.Button(i18n("btn_cache_latents"), variant="secondary")
                    cache_text_btn = gr.Button(i18n("btn_cache_text"), variant="secondary")
                    set_training_defaults_btn = gr.Button(i18n("btn_rec_params"), variant="secondary")

                with gr.Group():
                    gr.Markdown(i18n("header_basic_params"))
                    with gr.Row():
                        output_name = gr.Textbox(label=i18n("lbl_output_name"), value="my_lora", max_lines=1, interactive=True)
                        network_dim = gr.Number(label=i18n("lbl_network_dim"), value=16, interactive=True)
                        learning_rate = gr.Number(label=i18n("lbl_learning_rate"), value=1e-4, interactive=True)
                    with gr.Row():
                        max_train_epochs = gr.Number(label=i18n("lbl_max_train_epochs"), value=16, interactive=True)
                        save_every_n_epochs = gr.Number(label=i18n("lbl_save_every_n_epochs"), value=1, interactive=True)
                        discrete_flow_shift = gr.Number(label=i18n("lbl_discrete_flow_shift"), value=2.0, interactive=True)
                    with gr.Row():
                        optimizer_type = gr.Dropdown(label=i18n("lbl_optimizer_type"), choices=["AdamW", "AdamW64bit", "AdamW8bit", "Adafactor", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptLion", "DAdaptSGD", "Lion", "Lion8bit", "Prodigy", "RMSprop", "RMSprop8bit", "SGD", "SGD8bit"], value="AdamW8bit", interactive=True)
                        optimizer_args = gr.Textbox(label=i18n("lbl_optimizer_args"), placeholder="weight_decay=0.01", interactive=True)
                    with gr.Row():
                        lr_scheduler = gr.Dropdown(label=i18n("lbl_lr_scheduler"), choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], value="constant_with_warmup", interactive=True)
                        lr_warmup_steps = gr.Number(label=i18n("lbl_lr_warmup_steps"), value=0, interactive=True)
                        lr_decay_steps = gr.Number(label=i18n("lbl_lr_decay_steps"), value=0, interactive=True)
                    with gr.Row():
                        lr_scheduler_num_cycles = gr.Number(label=i18n("lbl_lr_scheduler_num_cycles"), value=1, interactive=True)
                        lr_scheduler_power = gr.Number(label=i18n("lbl_lr_scheduler_power"), value=1.0, interactive=True)
                        lr_scheduler_min_lr_ratio = gr.Number(label=i18n("lbl_lr_scheduler_min_lr_ratio"), value=0.0, interactive=True)

                with gr.Accordion(i18n("header_flow_timestep"), open=False):
                    with gr.Row():
                        weighting_scheme = gr.Dropdown(label=i18n("lbl_weighting_scheme"), choices=["none", "sigma_sqrt", "logit_normal", "mode"], value="none", interactive=True)
                        logit_mean = gr.Number(label=i18n("lbl_logit_mean"), value=0.0, interactive=True)
                        logit_std = gr.Number(label=i18n("lbl_logit_std"), value=1.0, interactive=True)
                    with gr.Row():
                        mode_scale = gr.Number(label=i18n("lbl_mode_scale"), value=1.29, interactive=True)
                        min_timestep = gr.Number(label=i18n("lbl_min_timestep"), value=0, interactive=True)
                        max_timestep = gr.Number(label=i18n("lbl_max_timestep"), value=1000, interactive=True)
                    with gr.Row():
                        preserve_distribution_shape = gr.Checkbox(label=i18n("lbl_preserve_distribution_shape"), value=False, interactive=True)
                        num_timestep_buckets = gr.Number(label=i18n("lbl_num_timestep_buckets"), value=0, interactive=True)

                with gr.Accordion(i18n("header_network_lora"), open=False):
                    with gr.Row():
                        network_alpha = gr.Number(label=i18n("lbl_network_alpha"), value=16, interactive=True)
                        network_dropout = gr.Number(label=i18n("lbl_network_dropout"), value=0.0, interactive=True)
                    with gr.Row():
                        network_args = gr.Textbox(label=i18n("lbl_network_args"), placeholder="key=val", interactive=True)
                        dim_from_weights = gr.Checkbox(label=i18n("lbl_dim_from_weights"), value=False, interactive=True)

                with gr.Accordion(i18n("header_training_flow"), open=False):
                    with gr.Row():
                        seed = gr.Number(label=i18n("lbl_seed"), value=42, interactive=True)
                        max_grad_norm = gr.Number(label=i18n("lbl_max_grad_norm"), value=1.0, interactive=True)
                        gradient_accumulation_steps = gr.Number(label=i18n("lbl_gradient_accumulation_steps"), value=1, interactive=True)
                    with gr.Row():
                        resume = gr.Textbox(label=i18n("lbl_resume"), placeholder="Path or HF ID", interactive=True)
                        save_state = gr.Checkbox(label=i18n("lbl_save_state"), value=False, interactive=True)
                        save_last_n_epochs = gr.Number(label=i18n("lbl_save_last_n_epochs"), value=None, interactive=True)
                    with gr.Row():
                        gradient_checkpointing_cpu_offload = gr.Checkbox(label=i18n("lbl_gradient_checkpointing_cpu_offload"), value=False, interactive=True)
                        disable_numpy_memmap = gr.Checkbox(label=i18n("lbl_disable_numpy_memmap"), value=False, interactive=True)

                with gr.Accordion(i18n("header_metadata"), open=False):
                    metadata_title = gr.Textbox(label=i18n("lbl_metadata_title"), interactive=True)
                    metadata_author = gr.Textbox(label=i18n("lbl_metadata_author"), interactive=True)
                    metadata_description = gr.Textbox(label=i18n("lbl_metadata_description"), interactive=True)
                    metadata_license = gr.Textbox(label=i18n("lbl_metadata_license"), interactive=True)
                    metadata_tags = gr.Textbox(label=i18n("lbl_metadata_tags"), interactive=True)

                with gr.Accordion(i18n("header_huggingface"), open=False):
                    with gr.Row():
                        huggingface_repo_id = gr.Textbox(label=i18n("lbl_huggingface_repo_id"), interactive=True)
                        huggingface_repo_type = gr.Dropdown(label=i18n("lbl_huggingface_repo_type"), choices=["model", "dataset"], value="model", interactive=True)
                    with gr.Row():
                        huggingface_path_in_repo = gr.Textbox(label=i18n("lbl_huggingface_path_in_repo"), interactive=True)
                        huggingface_token = gr.Textbox(label=i18n("lbl_huggingface_token"), interactive=True)
                        async_upload = gr.Checkbox(label=i18n("lbl_async_upload"), value=False, interactive=True)

                with gr.Accordion(i18n("header_sample_images"), open=False):
                    sample_images = gr.Checkbox(label=i18n("lbl_sample_images"), value=False, interactive=True)
                    with gr.Row():
                        sample_prompt = gr.Textbox(label=i18n("lbl_sample_prompt"), placeholder=i18n("ph_sample_prompt"), interactive=True)
                        sample_negative_prompt = gr.Textbox(label=i18n("lbl_sample_negative_prompt"), placeholder=i18n("ph_sample_negative_prompt"), interactive=True)
                    with gr.Row():
                        sample_w = gr.Number(label=i18n("lbl_sample_w"), value=1024, precision=0, interactive=True)
                        sample_h = gr.Number(label=i18n("lbl_sample_h"), value=1024, precision=0, interactive=True)
                        sample_every_n_epochs = gr.Number(label=i18n("lbl_sample_every_n_epochs"), value=1, precision=0, interactive=True)

                with gr.Accordion(i18n("lbl_additional_args"), open=False):
                    additional_args = gr.Textbox(label=i18n("lbl_additional_args"), placeholder=i18n("ph_additional_args"))

                start_training_btn = gr.Button(i18n("btn_start_training"), variant="primary")
                training_model_info = gr.Markdown("")
                training_status = gr.Markdown("")
                training_log = gr.Textbox(label="Training & Caching Log", lines=10, interactive=False)

            with gr.TabItem(i18n("tab_tools"), id="tools"):
                with gr.Group():
                    gr.Markdown("### Command Line Preview")
                    cmd_preview = gr.Code(label=i18n("lbl_cmd_preview"), language="shell", interactive=False)
                    copy_cmd_btn = gr.Button(i18n("btn_copy_cmd"))
                
                with gr.Group():
                    gr.Markdown("### Post-Processing")
                    with gr.Row():
                        input_lora = gr.Textbox(label=i18n("lbl_input_lora"), placeholder="Path to trained .safetensors file", max_lines=1)
                        output_comfy_lora = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder="Path to save converted model", max_lines=1)
                    
                    with gr.Row():
                        set_post_defaults_btn = gr.Button(i18n("btn_set_paths"), scale=0)
                        convert_btn = gr.Button(i18n("btn_convert"), variant="secondary")
                    
                    conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)

        def save_dataset_field_dynamic(state, value, index=None, field=None):
            if index is None or index >= len(state): return state
            new_state = list(state)
            ds = new_state[index]
            target_field = field
            is_video = hasattr(ds, "video_directory")
            if field == "jsonl_file": target_field = "video_jsonl_file" if is_video else "image_jsonl_file"
            
            try:
                if target_field in ["frame_stride", "max_frames", "frame_sample", "num_repeats", "batch_size", "fpack_latent_window_size"]:
                    value = int(value)
            except: pass
            
            if target_field == "target_frames":
                if isinstance(value, str):
                    try: value = [int(x.strip()) for x in value.split(",") if x.strip()]
                    except: pass
            if target_field in ["resolution", "fp_1f_clean_indices"]:
                if isinstance(value, str):
                    try: value = [int(x) for x in value.replace(",", " ").split()]
                    except: pass

            if hasattr(ds, target_field): setattr(ds, target_field, value)
            return new_state

        def remove_dataset_dynamic(state, index):
            manager = DatasetStateManager()
            manager.datasets = list(state)
            manager.remove_dataset(index)
            return manager.datasets

        def add_image_dataset(state, w, h, repeats, batch, cap_ext, eb, bnu):
            manager = DatasetStateManager()
            manager.datasets = list(state)
            manager.add_dataset("image")
            ds = manager.datasets[-1]
            ds.resolution = [int(w), int(h)]
            ds.num_repeats = int(repeats)
            ds.batch_size = int(batch)
            ds.caption_extension = cap_ext
            ds.enable_bucket = eb
            ds.bucket_no_upscale = bnu
            return manager.datasets

        def add_video_dataset(state, w, h, repeats, batch, cap_ext, eb, bnu):
            manager = DatasetStateManager()
            manager.datasets = list(state)
            manager.add_dataset("video")
            ds = manager.datasets[-1]
            ds.resolution = [int(w), int(h)]
            ds.num_repeats = int(repeats)
            ds.batch_size = int(batch)
            ds.caption_extension = cap_ext
            ds.enable_bucket = eb
            ds.bucket_no_upscale = bnu
            ds.target_frames = [1]
            return manager.datasets

        def validate_models_dir(path):
            if not path: return "Please enter ComfyUI models path."
            required = ["diffusion_models", "vae", "text_encoders"]
            missing = [d for d in required if not os.path.exists(os.path.join(path, d))]
            if missing: return f"Missing: {', '.join(missing)}"
            return "Valid ComfyUI models directory found."

        def set_recommended_settings(project_path, model_arch, vram_val):
            w, h = config_manager.get_resolution(model_arch)
            bs = config_manager.get_batch_size(model_arch, vram_val)
            if project_path: 
                save_project_settings(project_path, resolution_w=w, resolution_h=h, batch_size=bs)
            return w, h, bs

        def set_preprocessing_defaults(project_path, comfy_models_dir, model_arch):
            if not comfy_models_dir: return gr.update(), gr.update(), gr.update()
            vae, te1, te2 = config_manager.get_preprocessing_paths(model_arch, comfy_models_dir)
            if project_path:
                save_project_settings(project_path, vae_path=vae, text_encoder1_path=te1, text_encoder2_path=te2)
            return vae, te1, te2

        def set_post_processing_defaults(project_path, output_nm):
            if not project_path or not output_nm: return gr.update(), gr.update()
            models_dir = os.path.join(project_path, "models")
            in_lora = os.path.join(models_dir, f"{output_nm}.safetensors")
            out_lora = os.path.join(models_dir, f"{output_nm}_comfy.safetensors")
            save_project_settings(project_path, input_lora_path=in_lora, output_comfy_lora_path=out_lora)
            return in_lora, out_lora
            
        def run_command(command):
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                log = command + "\n\n"
                for line in process.stdout:
                    log += line
                    yield log
                process.wait()
                yield log + ("\nDone." if process.returncode == 0 else f"\nError {process.returncode}")
            except Exception as e: yield f"Error: {e}"

        def save_project_settings(project_path, **kwargs):
            try:
                settings_path = os.path.join(project_path, "musubi_project.toml")
                settings = {}
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = toml.load(f)
                settings.update(kwargs)
                with open(settings_path, "w", encoding="utf-8") as f:
                    toml.dump(settings, f)
            except Exception as e:
                print(f"Error saving project settings: {e}")

        def load_project_settings(project_path):
            settings = {}
            try:
                settings_path = os.path.join(project_path, "musubi_project.toml")
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        settings = toml.load(f)
            except: pass
            return settings

        def init_project(path):
            if not path:
                return ["Please enter a project directory path."] + [gr.update()] * 80
            try:
                os.makedirs(os.path.join(path, "training"), exist_ok=True)
                settings = load_project_settings(path)
                def s(k, d=None): return settings.get(k, d)

                res = [
                    f"Project initialized at {path}.",
                    s("model_arch", "Wan 2.1 (T2V-14B)"),
                    s("vram_size", "24"),
                    s("comfy_models_dir", ""),
                    s("resolution_w", 1024),
                    s("resolution_h", 1024),
                    s("num_repeats", 1),
                    s("batch_size", 1),
                    # toml_preview loaded below
                    "", 
                    s("vae_path", ""),
                    s("text_encoder1_path", ""),
                    s("text_encoder2_path", ""),
                    s("dit_path", ""),
                    s("output_name", "my_lora"),
                    s("network_dim", 16),
                    s("learning_rate", 1e-4),
                    s("max_train_epochs", 16),
                    s("save_every_n_epochs", 1),
                    s("discrete_flow_shift", 2.0),
                    s("blocks_to_swap", 0),
                    s("use_pinned_memory_for_block_swap", False),
                    s("mixed_precision", "bf16"),
                    s("gradient_checkpointing", True),
                    s("fp8_scaled", True),
                    s("fp8_llm", True),
                    s("additional_args", ""),
                    s("sample_images", False),
                    s("sample_every_n_epochs", 1),
                    s("sample_prompt", ""),
                    s("sample_negative_prompt", ""),
                    s("sample_w", 1024),
                    s("sample_h", 1024),
                    s("input_lora_path", ""),
                    s("output_comfy_lora_path", ""),
                    s("caption_extension", ".txt"),
                    s("enable_bucket", False),
                    s("bucket_no_upscale", False),
                    
                    s("optimizer_type", "AdamW8bit"), s("optimizer_args", ""),
                    s("lr_scheduler", "constant_with_warmup"), s("lr_warmup_steps", 0), s("lr_decay_steps", 0),
                    s("lr_scheduler_num_cycles", 1), s("lr_scheduler_power", 1.0), s("lr_scheduler_min_lr_ratio", 0.0),
                    
                    s("weighting_scheme", "none"), s("logit_mean", 0.0), s("logit_std", 1.0),
                    s("mode_scale", 1.29), s("min_timestep", 0), s("max_timestep", 1000),
                    s("preserve_distribution_shape", False), s("num_timestep_buckets", 0),
                    
                    s("network_alpha", 16), s("network_dropout", 0.0), s("network_args", ""), s("dim_from_weights", False),
                    
                    s("seed", 42), s("max_grad_norm", 1.0), s("gradient_accumulation_steps", 1),
                    s("resume", ""), s("save_state", False), s("save_last_n_epochs", None),
                    s("gradient_checkpointing_cpu_offload", False), s("disable_numpy_memmap", False),
                    s("offload_txt_in", False), s("attn_mode", "sdpa"),
                    
                    s("metadata_title", ""), s("metadata_author", ""), s("metadata_description", ""), s("metadata_license", ""), s("metadata_tags", ""),
                    
                    s("huggingface_repo_id", ""), s("huggingface_repo_type", "model"), s("huggingface_path_in_repo", ""), s("huggingface_token", ""), s("async_upload", False),
                ]

                loaded_state = []
                preview_content = ""
                config_path = os.path.join(path, "dataset_config.toml")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            preview_content = f.read()
                        manager = DatasetStateManager()
                        manager.from_toml_string(preview_content)
                        loaded_state = manager.datasets
                    except: pass
                
                res[8] = preview_content 
                res.append(loaded_state if loaded_state else [])
                return res
            except Exception as e:
                return [f"Error initializing project: {str(e)}"] + [gr.update()] * 80

        def set_training_defaults(project_path, comfy_models_dir, model_arch, vram_val):
            defaults = config_manager.get_training_defaults(model_arch, vram_val, comfy_models_dir)
            
            dit = defaults.get("dit_path", "")
            dim = defaults.get("network_dim", 16)
            lr = defaults.get("learning_rate", 1e-4)
            epochs = 16 
            save_n = 1
            flow = defaults.get("discrete_flow_shift", 2.0)
            
            swap = defaults.get("blocks_to_swap", defaults.get("block_swap", 0))
            pinned = False
            prec = defaults.get("mixed_precision", "bf16")
            grad_cp = defaults.get("gradient_checkpointing", True)
            f8s = defaults.get("fp8_scaled", True)
            f8l = defaults.get("fp8_llm", True)

            return (
                dit, dim, lr, epochs, save_n, flow,
                swap, pinned, prec, grad_cp, f8s, f8l,
                1, 1024, 1024,
                "AdamW8bit", "",
                "constant_with_warmup", 0, 0,
                1, 1.0, 0.0,
                "none", 0.0, 1.0,
                1.29, 0, 1000,
                False, 0,
                16, 0.0, "", False,
                42, 1.0, 1,
                "", False, None,
                False, False,
                False, "sdpa",
                "", "", "", "", "",
                "", "model", "", "", False
            )

        def cache_latents(project_path, vae, te1, te2, model, comfy, w, h, batch, vram):
            if not project_path: yield "Error: No project."; return
            save_project_settings(project_path, model_arch=model, comfy_models_dir=comfy, resolution_w=w, resolution_h=h, batch_size=batch, vae_path=vae, text_encoder1_path=te1, text_encoder2_path=te2)
            if not vae: yield "Error: No VAE."; return
            config = os.path.join(project_path, "dataset_config.toml")
            script = "qwen_image_cache_latents.py" if model == "Qwen-Image" else "zimage_cache_latents.py"
            cmd = f"{sys.executable} src/musubi_tuner/{script} --dataset_config {config} --vae {vae}"
            yield from run_command(cmd)

        def cache_text_encoder(project_path, te1, te2, vae, model, comfy, w, h, batch, vram):
            if not project_path: yield "Error: No project."; return
            save_project_settings(project_path, model_arch=model, comfy_models_dir=comfy, resolution_w=w, resolution_h=h, batch_size=batch, vae_path=vae, text_encoder1_path=te1, text_encoder2_path=te2)
            if not te1: yield "Error: No TE1."; return
            config = os.path.join(project_path, "dataset_config.toml")
            script = "qwen_image_cache_text_encoder_outputs.py" if model == "Qwen-Image" else "zimage_cache_text_encoder_outputs.py"
            cmd = f"{sys.executable} src/musubi_tuner/{script} --dataset_config {config} --text_encoder {te1} --batch_size 1"
            if model == "Qwen-Image" and vram in ["12", "16"]: cmd += " --fp8_vl"
            yield from run_command(cmd)

        def generate_config(dataset_state_vals, project_path, w, h, repeats, batch, cap_ext, enable_buck, buck_no_up, model_val, vram_val, comfy_val, vae_val, te1_val, te2_val):
            if not project_path: return "Error: Project directory not specified.", ""
            save_project_settings(project_path, model_arch=model_val, vram_size=vram_val, comfy_models_dir=comfy_val, resolution_w=w, resolution_h=h, num_repeats=repeats, batch_size=batch, vae_path=vae_val, text_encoder1_path=te1_val, text_encoder2_path=te2_val, caption_extension=cap_ext, enable_bucket=enable_buck, bucket_no_upscale=buck_no_up)
            
            manager = DatasetStateManager()
            if dataset_state_vals: manager.datasets = dataset_state_vals
            else: manager.add_dataset("image")
            
            manager.update_general_params({
                "resolution": [int(w), int(h)], "num_repeats": int(repeats), "batch_size": int(batch),
                "caption_extension": cap_ext, "enable_bucket": enable_buck, "bucket_no_upscale": buck_no_up
            })
            
            toml_str = manager.to_toml_string(architecture=model_val)
            config_path = os.path.join(project_path, "dataset_config.toml")
            try:
                with open(config_path, "w", encoding="utf-8") as f: f.write(toml_str)
                status_msg = f"✅ Dataset config saved to: {config_path}"
            except Exception as e:
                status_msg = f"❌ Error saving config: {str(e)}"
                toml_str = f"# Error: {str(e)}"
            return status_msg, gr.update(value=toml_str, visible=True)

        def convert_lora_to_comfy(project_path, input_path, output_path, model, comfy, w, h, batch, vae, te1, te2):
            if not project_path: yield "Error: Project directory not set."; return
            save_project_settings(project_path, model_arch=model, comfy_models_dir=comfy, resolution_w=w, resolution_h=h, batch_size=batch, vae_path=vae, text_encoder1_path=te1, text_encoder2_path=te2, input_lora_path=input_path, output_comfy_lora_path=output_path)
            if not input_path or not output_path: yield "Error: Paths required."; return
            script_path = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
            cmd = f"{sys.executable} {script_path} {input_path} {output_path}"
            yield from run_command(cmd)

        def generate_cmd_preview(
            project_path, model, output_nm, dim, lr, epochs, save_n, flow_shift, swap, use_pinned_memory_for_block_swap,
            prec, grad_cp, fp8_s_generic, fp8_l_generic, add_args,
            should_sample_images, sample_every_n, sample_prompt_val, sample_negative_prompt_val, sample_w_val, sample_h_val,
            wan_t5, wan_clip, wan_fp8_s, wan_fp8_t5, wan_vae_cpu, wan_1f,
            wan_dit_hn, wan_ts_bound, wan_offload, wan_force_v21,
            hv_dit, hv_te1, hv_te2, hv_vae, hv_fp8_l, hv_fp8_s, hv_vae_tiling, hv_vae_chunk,
            hv15_task, hv15_dit, hv15_master_te, hv15_byt5, hv15_img_enc, hv15_fp8_vl, hv15_fp8_s, hv15_vae_s,
            fpack_te1, fpack_te2, fpack_img_enc, fpack_lw, fpack_bd, fpack_f1, fpack_1f,
            flux_te1, flux_te2, flux_fp8_t5, flux_fp8_s,
            k5_task, k5_sched, k5_nabla,
            qwen_te, qwen_fp8_vl, qwen_fp8_s,
            zimage_te, zimage_fp8_l, zimage_fp8_s, zimage_32,
            dit_basic, vae_basic, te1_basic,
            opt_type, opt_args, sched, warmup, decay, cycles, power, min_lr_ratio,
            weighting, logit_m, logit_s, m_scale, min_ts, max_ts, pres_dist, ts_buckets,
            n_alpha, n_dropout, n_args, dim_fw,
            seed_val, max_norm, grad_acc, resume_val, save_st, save_last,
            grad_cp_cpu, no_memmap, offload_in, attn,
            m_title, m_author, m_desc, m_license, m_tags,
            hf_repo, hf_type, hf_path, hf_token, hf_async
        ):
            if not project_path: return "Project directory not set."
            
            dataset_config = os.path.join(project_path, "dataset_config.toml")
            output_dir = os.path.join(project_path, "models")
            logging_dir = os.path.join(project_path, "logs")
            script_name = ""
            script_args = []

            if "Wan" in model:
                script_name = "wan_train_network.py"
                task_map = {"Wan 2.1 (T2V-14B)": "t2v-14B", "Wan 2.2 (T2V-14B)": "t2v-14B", "Wan 2.2 (T2V-5B)": "t2v-5B"}
                script_args = ["--task", task_map.get(model, "t2v-14B"), "--dit", dit_basic]
                if wan_t5: script_args.extend(["--t5", wan_t5])
                if wan_clip: script_args.extend(["--clip", wan_clip])
                if wan_fp8_s: script_args.append("--fp8_scaled")
                if wan_fp8_t5: script_args.append("--fp8_t5")
                if wan_vae_cpu: script_args.append("--vae_cache_cpu")
                if wan_1f: script_args.append("--one_frame")
                if "Wan 2.2" in model:
                    if wan_dit_hn: script_args.extend(["--dit_high_noise", wan_dit_hn])
                    if wan_ts_bound: script_args.extend(["--timestep_boundary", str(wan_ts_bound)])
                    if wan_offload: script_args.append("--offload_inactive_dit")
                    if wan_force_v21: script_args.append("--force_v2_1_time_embedding")
            elif model == "HunyuanVideo":
                script_name = "hv_train_network.py"
                script_args = ["--dit", hv_dit, "--vae", hv_vae, "--text_encoder1", hv_te1, "--text_encoder2", hv_te2]
                if hv_fp8_l: script_args.append("--fp8_llm")
                if hv_fp8_s: script_args.append("--fp8_scaled")
                if hv_vae_tiling: script_args.append("--vae_tiling")
                if hv_vae_chunk: script_args.extend(["--vae_chunk_size", str(int(hv_vae_chunk))])
            elif model == "Hunyuan 1.5":
                script_name = "hv_1_5_train_network.py"
                script_args = ["--task", hv15_task, "--dit", hv15_dit, "--text_encoder", hv15_master_te, "--byt5", hv15_byt5, "--image_encoder", hv15_img_enc]
                if hv15_fp8_vl: script_args.append("--fp8_vl")
                if hv15_fp8_s: script_args.append("--fp8_scaled")
                if hv15_vae_s: script_args.extend(["--vae_sample_size", str(int(hv15_vae_s))])
            elif model == "FramePack":
                script_name = "fpack_train_network.py"
                script_args = ["--dit", dit_basic, "--text_encoder1", fpack_te1, "--text_encoder2", fpack_te2, "--image_encoder", fpack_img_enc]
                if fpack_lw: script_args.extend(["--latent_window_size", str(int(fpack_lw))])
                if fpack_bd: script_args.append("--bulk_decode")
                if fpack_f1: script_args.append("--f1")
                if fpack_1f: script_args.append("--one_frame")
            elif model == "Flux.1 Kontext":
                script_name = "flux_kontext_train_network.py"
                script_args = ["--dit", dit_basic, "--text_encoder1", flux_te1, "--text_encoder2", flux_te2]
                if flux_fp8_t5: script_args.append("--fp8_t5")
                if flux_fp8_s: script_args.append("--fp8_scaled")
            elif model == "Kandinsky 5":
                script_name = "kandinsky5_train_network.py"
                script_args = ["--task", k5_task, "--dit", dit_basic]
                if k5_sched: script_args.extend(["--scheduler_scale", str(k5_sched)])
                if k5_nabla: script_args.append("--force_nabla_attention")
            elif model == "Qwen-Image":
                script_name = "qwen_image_train_network.py"
                script_args = ["--dit", dit_basic, "--text_encoder", qwen_te]
                if qwen_fp8_vl: script_args.append("--fp8_vl")
                if qwen_fp8_s: script_args.append("--fp8_scaled")
            elif model == "Z-Image-Turbo":
                script_name = "zimage_train_network.py"
                script_args = ["--dit", dit_basic, "--text_encoder", zimage_te]
                if zimage_fp8_l: script_args.append("--fp8_llm")
                if zimage_fp8_s: script_args.append("--fp8_scaled")
                if zimage_32: script_args.append("--use_32bit_attention")

            if not script_name: return "Unknown model."

            cmd = ["accelerate", "launch", os.path.join("src", "musubi_tuner", script_name)]
            cmd.extend([str(x) for x in script_args if x is not None])

            cmd.extend([
                "--dataset_config", dataset_config, "--output_dir", output_dir, "--output_name", output_nm,
                "--network_dim", str(int(dim)), "--learning_rate", str(lr),
                "--max_train_epochs", str(int(epochs)), "--save_every_n_epochs", str(int(save_n)),
                "--discrete_flow_shift", str(flow_shift), "--mixed_precision", prec,
                "--logging_dir", logging_dir, "--log_with", "tensorboard",
                "--seed", str(int(seed_val)), "--max_grad_norm", str(max_norm),
                "--gradient_accumulation_steps", str(int(grad_acc)),
                "--optimizer_type", opt_type, "--lr_scheduler", sched,
                "--lr_warmup_steps", str(int(warmup)), "--lr_scheduler_num_cycles", str(int(cycles)),
                "--lr_scheduler_power", str(power), "--network_alpha", str(int(n_alpha))
            ])
            
            if opt_args: cmd.extend(["--optimizer_args", opt_args])
            if decay: cmd.extend(["--lr_decay_steps", str(int(decay))])
            if min_lr_ratio: cmd.extend(["--lr_scheduler_min_lr_ratio", str(min_lr_ratio)])
            if weighting != "none": cmd.extend(["--weighting_scheme", weighting])
            if logit_m != 0.0: cmd.extend(["--logit_mean", str(logit_m)])
            if logit_s != 1.0: cmd.extend(["--logit_std", str(logit_s)])
            if weighting == "mode": cmd.extend(["--mode_scale", str(m_scale)])
            if min_ts > 0: cmd.extend(["--min_timestep", str(int(min_ts))])
            if max_ts < 1000: cmd.extend(["--max_timestep", str(int(max_ts))])
            if pres_dist: cmd.append("--preserve_distribution_shape")
            if ts_buckets: cmd.extend(["--num_timestep_buckets", str(int(ts_buckets))])
            if n_dropout > 0: cmd.extend(["--network_dropout", str(n_dropout)])
            if n_args: cmd.extend(["--network_args", n_args])
            if dim_fw: cmd.append("--dim_from_weights")
            if resume_val: cmd.extend(["--resume", resume_val])
            if save_st: cmd.append("--save_state")
            if save_last is not None: cmd.extend(["--save_last_n_epochs", str(int(save_last))])
            if grad_cp: cmd.append("--gradient_checkpointing")
            if grad_cp_cpu: cmd.append("--gradient_checkpointing_cpu_offload")
            if no_memmap: cmd.append("--disable_numpy_memmap")
            if offload_in: cmd.append("--offload_txt_in")
            
            if attn == "flash_attn": cmd.append("--flash_attn")
            elif attn == "sage_attn": cmd.append("--sage_attn")
            elif attn == "xformers": cmd.append("--xformers")
            elif attn == "split_attn": cmd.append("--split_attn")
            else: cmd.append("--sdpa")
            
            if m_title: cmd.extend(["--metadata_title", m_title])
            if m_author: cmd.extend(["--metadata_author", m_author])
            if m_desc: cmd.extend(["--metadata_description", m_desc])
            if m_license: cmd.extend(["--metadata_license", m_license])
            if m_tags: cmd.extend(["--metadata_tags", m_tags])
            
            if hf_repo:
                cmd.extend(["--huggingface_repo_id", hf_repo, "--huggingface_repo_type", hf_type])
                if hf_path: cmd.extend(["--huggingface_path_in_repo", hf_path])
                if hf_token: cmd.extend(["--huggingface_token", hf_token])
                if hf_async: cmd.append("--async_upload")
            
            if swap > 0:
                cmd.extend(["--blocks_to_swap", str(int(swap))])
                if use_pinned_memory_for_block_swap: cmd.append("--use_pinned_memory_for_block_swap")
            
            if add_args:
                try: cmd.extend(shlex.split(add_args))
                except: pass

            return subprocess.list2cmdline([str(x) for x in cmd])

        def start_training(*args):
            project_dir_val = args[0]
            cmd_str = generate_cmd_preview(*args)

            save_project_settings(
                project_dir_val,
                model_arch=args[1], output_name=args[2], network_dim=args[3], learning_rate=args[4],
                max_train_epochs=args[5], save_every_n_epochs=args[6], discrete_flow_shift=args[7],
                blocks_to_swap=args[8], use_pinned_memory_for_block_swap=args[9],
                mixed_precision=args[10], gradient_checkpointing=args[11],
                fp8_scaled=args[12], fp8_llm=args[13], additional_args=args[14],
            )

            try:
                final_cmd = f"{cmd_str} & echo. & echo Training finished. & pause >nul"
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                subprocess.Popen(["cmd", "/c", final_cmd], creationflags=flags, shell=False)
                return f"Training started!\nCommand: {cmd_str}"
            except Exception as e:
                return f"Error: {e}"

        preview_inputs = [
            project_dir, model_arch, output_name, network_dim, learning_rate, max_train_epochs, save_every_n_epochs, discrete_flow_shift, blocks_to_swap, use_pinned_memory_for_block_swap,
            mixed_precision, gradient_checkpointing, fp8_scaled, fp8_llm, additional_args,
            sample_images, sample_every_n_epochs, sample_prompt, sample_negative_prompt, sample_w, sample_h,
            wan_t5, wan_clip, wan_fp8_scaled, wan_fp8_t5, wan_vae_cache_cpu, wan_one_frame,
            wan_dit_high_noise, wan_timestep_boundary, wan_offload_inactive_dit, wan_force_v2_1_time_embedding,
            hv_dit, hv_te1, hv_te2, hv_vae, hv_fp8_llm, hv_fp8_scaled, hv_vae_tiling, hv_vae_chunk_size,
            hv15_task, hv15_dit, hv15_master_te, hv15_byt5, hv15_img_enc, hv15_fp8_vl, hv15_fp8_scaled, hv15_vae_sample_size,
            fpack_te1, fpack_te2, fpack_img_enc, fpack_latent_window_size, fpack_bulk_decode, fpack_f1, fpack_one_frame,
            flux_te1, flux_te2, flux_fp8_t5, flux_fp8_scaled,
            k5_task, k5_scheduler_scale, k5_force_nabla_attention,
            qwen_te, qwen_fp8_vl, qwen_fp8_scaled,
            zimage_te, zimage_fp8_llm, zimage_fp8_scaled, zimage_use_32bit_attention,
            dit_path, vae_path, text_encoder1_path,
            optimizer_type, optimizer_args, lr_scheduler, lr_warmup_steps, lr_decay_steps, lr_scheduler_num_cycles, lr_scheduler_power, lr_scheduler_min_lr_ratio,
            weighting_scheme, logit_mean, logit_std, mode_scale, min_timestep, max_timestep, preserve_distribution_shape, num_timestep_buckets,
            network_alpha, network_dropout, network_args, dim_from_weights,
            seed, max_grad_norm, gradient_accumulation_steps, resume, save_state, save_last_n_epochs,
            gradient_checkpointing_cpu_offload, disable_numpy_memmap, offload_txt_in, attn_mode,
            metadata_title, metadata_author, metadata_description, metadata_license, metadata_tags,
            huggingface_repo_id, huggingface_repo_type, huggingface_path_in_repo, huggingface_token, async_upload
        ]

        init_btn.click(
            fn=init_project,
            inputs=[project_dir],
            outputs=[
                project_status, model_arch, vram_size, comfy_models_dir, resolution_w, resolution_h, num_repeats, batch_size, toml_preview, 
                vae_path, text_encoder1_path, text_encoder2_path, dit_path,
                output_name, network_dim, learning_rate, max_train_epochs, save_every_n_epochs, discrete_flow_shift, blocks_to_swap, use_pinned_memory_for_block_swap,
                mixed_precision, gradient_checkpointing, fp8_scaled, fp8_llm, additional_args,
                sample_images, sample_every_n_epochs, sample_prompt, sample_negative_prompt, sample_w, sample_h,
                input_lora, output_comfy_lora, caption_extension, enable_bucket, bucket_no_upscale,
                optimizer_type, optimizer_args, lr_scheduler, lr_warmup_steps, lr_decay_steps, lr_scheduler_num_cycles, lr_scheduler_power, lr_scheduler_min_lr_ratio,
                weighting_scheme, logit_mean, logit_std, mode_scale, min_timestep, max_timestep, preserve_distribution_shape, num_timestep_buckets,
                network_alpha, network_dropout, network_args, dim_from_weights,
                seed, max_grad_norm, gradient_accumulation_steps, resume, save_state, save_last_n_epochs,
                gradient_checkpointing_cpu_offload, disable_numpy_memmap, offload_txt_in, attn_mode,
                metadata_title, metadata_author, metadata_description, metadata_license, metadata_tags,
                huggingface_repo_id, huggingface_repo_type, huggingface_path_in_repo, huggingface_token, async_upload,
                dataset_state
            ]
        )

        add_img_ds_btn.click(fn=add_image_dataset, inputs=[dataset_state, resolution_w, resolution_h, num_repeats, batch_size, caption_extension, enable_bucket, bucket_no_upscale], outputs=[dataset_state])
        add_vid_ds_btn.click(fn=add_video_dataset, inputs=[dataset_state, resolution_w, resolution_h, num_repeats, batch_size, caption_extension, enable_bucket, bucket_no_upscale], outputs=[dataset_state])

        gen_toml_btn.click(fn=generate_config, inputs=[dataset_state, project_dir, resolution_w, resolution_h, num_repeats, batch_size, caption_extension, enable_bucket, bucket_no_upscale, model_arch, vram_size, comfy_models_dir, vae_path, text_encoder1_path, text_encoder2_path], outputs=[dataset_status, toml_preview])
        
        validate_models_btn.click(fn=validate_models_dir, inputs=[comfy_models_dir], outputs=[models_status])
        
        rec_params_btn.click(fn=set_recommended_settings, inputs=[project_dir, model_arch, vram_size], outputs=[resolution_w, resolution_h, batch_size])
        set_rec_settings_btn.click(fn=set_recommended_settings, inputs=[project_dir, model_arch, vram_size], outputs=[resolution_w, resolution_h, batch_size])
        
        set_paths_btn.click(fn=set_preprocessing_defaults, inputs=[project_dir, comfy_models_dir, model_arch], outputs=[vae_path, text_encoder1_path, text_encoder2_path])
        
        set_post_defaults_btn.click(fn=set_post_processing_defaults, inputs=[project_dir, output_name], outputs=[input_lora, output_comfy_lora])

        set_training_defaults_btn.click(
            fn=set_training_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch, vram_size],
            outputs=[
                dit_path, network_dim, learning_rate, max_train_epochs, save_every_n_epochs, discrete_flow_shift, blocks_to_swap, use_pinned_memory_for_block_swap,
                mixed_precision, gradient_checkpointing, fp8_scaled, fp8_llm, 
                sample_every_n_epochs, sample_w, sample_h,
                optimizer_type, optimizer_args, lr_scheduler, lr_warmup_steps, lr_decay_steps, lr_scheduler_num_cycles, lr_scheduler_power, lr_scheduler_min_lr_ratio,
                weighting_scheme, logit_mean, logit_std, mode_scale, min_timestep, max_timestep, preserve_distribution_shape, num_timestep_buckets,
                network_alpha, network_dropout, network_args, dim_from_weights,
                seed, max_grad_norm, gradient_accumulation_steps, resume, save_state, save_last_n_epochs,
                gradient_checkpointing_cpu_offload, disable_numpy_memmap, offload_txt_in, attn_mode,
                metadata_title, metadata_author, metadata_description, metadata_license, metadata_tags,
                huggingface_repo_id, huggingface_repo_type, huggingface_path_in_repo, huggingface_token, async_upload
            ]
        )

        cache_latents_btn.click(fn=cache_latents, inputs=[project_dir, vae_path, text_encoder1_path, text_encoder2_path, model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size, vram_size], outputs=[training_log])
        cache_text_btn.click(fn=cache_text_encoder, inputs=[project_dir, text_encoder1_path, text_encoder2_path, vae_path, model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size, vram_size], outputs=[training_log])
        
        start_training_btn.click(fn=start_training, inputs=preview_inputs, outputs=[training_status])
        
        convert_btn.click(fn=convert_lora_to_comfy, inputs=[project_dir, input_lora, output_comfy_lora, model_arch, comfy_models_dir, resolution_w, resolution_h, batch_size, vae_path, text_encoder1_path, text_encoder2_path], outputs=[conversion_log])

        for component in preview_inputs:
             if hasattr(component, "change"):
                 component.change(fn=generate_cmd_preview, inputs=preview_inputs, outputs=[cmd_preview])

        copy_cmd_btn.click(fn=None, inputs=[cmd_preview], outputs=None, js="(x) => { navigator.clipboard.writeText(x); alert('Command copied to clipboard!'); }")

    return demo

if __name__ == "__main__":
    with gr.Blocks(title="Musubi Tuner GUI", css=load_custom_css()) as demo:
        construct_ui()
    demo.launch()