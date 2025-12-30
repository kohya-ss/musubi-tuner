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
        with open(css_path, "r", encoding="utf-8") as f: return f.read()
    except FileNotFoundError: return ""

I18N_CACHE = None
CURRENT_LANG = None

def load_i18n_data():
    global I18N_CACHE
    if I18N_CACHE is not None: return I18N_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_data.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f: I18N_CACHE = json.load(f)
    except: I18N_CACHE = {"en": {}}
    return I18N_CACHE

def i18n(key):
    global CURRENT_LANG
    if not CURRENT_LANG:
        lang = "en"
        try:
            env_lang = os.environ.get("LANG", "").lower()
            if env_lang.startswith("ja"): lang = "ja"
            else:
                sys_lang = locale.getdefaultlocale()[0]
                if sys_lang and sys_lang.lower().startswith("ja"): lang = "ja"
        except: pass
        CURRENT_LANG = lang
    data = load_i18n_data()
    return data.get(CURRENT_LANG, {}).get(key, data.get("en", {}).get(key, key))

def create_component(arg_def):
    name = arg_def["name"]
    label_key = arg_loader.get_label_key(name)
    label = i18n(label_key)
    val = arg_def.get("default")
    
    if arg_def["type"] == "text":
        return gr.Textbox(label=label, value=val if val is not None else "", max_lines=1)
    elif arg_def["type"] == "number":
        return gr.Number(label=label, value=val, precision=None)
    elif arg_def["type"] == "checkbox":
        return gr.Checkbox(label=label, value=val if val is not None else False)
    elif arg_def["type"] == "dropdown":
        return gr.Dropdown(label=label, choices=arg_def.get("choices", []), value=val)
    elif arg_def["type"] == "slider":
        return gr.Slider(label=label, minimum=arg_def.get("min", 0), maximum=arg_def.get("max", 100), value=val if val is not None else 0, step=1)
    return gr.Textbox(label=label)

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

# --- Main UI Construction ---

def construct_ui():
    custom_css = load_custom_css()
    ui_components = {}
    model_arg_groups = {}
    arg_defs_map = arg_loader.get_all_args_map()

    with gr.Group(elem_classes=["main-container"]) as demo:
        gr.Markdown(i18n("app_header"))
        gr.Markdown(i18n("app_desc"))

        with gr.Tabs() as tabs:
            with gr.TabItem(i18n("tab_init"), id="init"):
                gr.Markdown(i18n("desc_project"))
                with gr.Row():
                    project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"), max_lines=1)
                    ui_components["project_dir"] = project_dir 
                
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
                        value="Wan 2.2 (T2V-14B)", interactive=True
                    )
                    ui_components["model_arch"] = model_arch
                    vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24", interactive=True)
                    ui_components["vram_size"] = vram_size

                with gr.Group():
                    gr.Markdown("### ComfyUI & Paths")
                    with gr.Row():
                        comfy_models_dir = gr.Textbox(label=i18n("lbl_comfy_dir"), placeholder=i18n("ph_comfy_dir"), max_lines=1, interactive=True)
                        ui_components["comfy_models_dir"] = comfy_models_dir
                        validate_models_btn = gr.Button(i18n("btn_validate_models"), scale=0)
                    models_status = gr.Markdown("")

                    with gr.Row():
                        ui_components["dit_path"] = gr.Textbox(label=i18n("lbl_dit"), max_lines=1, interactive=True)
                        ui_components["vae_path"] = gr.Textbox(label=i18n("lbl_vae"), max_lines=1, interactive=True)
                    with gr.Row():
                        ui_components["text_encoder1_path"] = gr.Textbox(label=i18n("lbl_te1"), max_lines=1, interactive=True)
                        ui_components["text_encoder2_path"] = gr.Textbox(label=i18n("lbl_te2"), max_lines=1, interactive=True)

                    model_specs = arg_loader.get_model_specific_args_map()
                    for group_key, group_data in model_specs.items():
                        with gr.Group(visible=False) as g:
                            model_arg_groups[group_key] = g
                            if "group_name" in group_data:
                                gr.Markdown(f"### {group_data['group_name']}")

                            args_list = group_data.get("args", [])
                            for i in range(0, len(args_list), 2):
                                with gr.Row():
                                    for j in range(2):
                                        if i + j < len(args_list):
                                            arg = args_list[i+j]
                                            comp = create_component(arg)
                                            ui_components[arg["name"]] = comp
                    
                    with gr.Row():
                        set_paths_btn = gr.Button(i18n("btn_set_paths"))
                        rec_params_btn = gr.Button(i18n("btn_rec_params"))

                with gr.Accordion("Performance & Memory", open=False):
                    perf_essential = arg_loader.get_essential_performance_args()
                    with gr.Row():
                        for arg in perf_essential:
                            ui_components[arg["name"]] = create_component(arg)
                    
                    perf_advanced = arg_loader.get_advanced_performance_args()
                    if perf_advanced:
                        with gr.Row():
                             for arg in perf_advanced:
                                 ui_components[arg["name"]] = create_component(arg)

            with gr.TabItem(i18n("tab_dataset"), id="dataset"):
                gr.Markdown(i18n("desc_dataset"))
                dataset_state = gr.State([])
                with gr.Row():
                    resolution_w = gr.Number(label=i18n("lbl_res_w"), value=1024, precision=0)
                    resolution_h = gr.Number(label=i18n("lbl_res_h"), value=1024, precision=0)
                    ui_components["resolution_w"] = resolution_w
                    ui_components["resolution_h"] = resolution_h
                    
                    num_repeats = gr.Number(label=i18n("lbl_num_repeats"), value=1, precision=0)
                    ui_components["num_repeats"] = num_repeats
                    
                    batch_size = gr.Number(label=i18n("lbl_batch_size"), value=1, precision=0)
                    ui_components["batch_size"] = batch_size
                    
                    set_rec_settings_btn = gr.Button(i18n("btn_rec_res_batch"))

                with gr.Row():
                    ds_opts = arg_loader.get_essential_dataset_options()
                    for arg in ds_opts:
                        if arg["name"] in ["caption_extension"]:
                             ui_components[arg["name"]] = create_component(arg)

                    ui_components["enable_bucket"] = gr.Checkbox(label=i18n("lbl_enable_bucket"), value=False)
                    ui_components["bucket_no_upscale"] = gr.Checkbox(label=i18n("lbl_bucket_no_upscale"), value=False)

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
                    ess_args = arg_loader.get_essential_training_args()
                    for i in range(0, len(ess_args), 3):
                        with gr.Row():
                            for j in range(3):
                                if i+j < len(ess_args):
                                    arg = ess_args[i+j]
                                    ui_components[arg["name"]] = create_component(arg)

                adv_args_groups = arg_loader.get_advanced_training_args()
                for group_name, group_args in adv_args_groups.items():
                    header_key = f"header_{group_name}"
                    header_text = i18n(header_key)
                    if header_text == header_key: header_text = group_name.replace("_", " ").title()

                    with gr.Accordion(header_text, open=False):
                        for i in range(0, len(group_args), 3):
                            with gr.Row():
                                for j in range(3):
                                    if i+j < len(group_args):
                                        arg = group_args[i+j]
                                        ui_components[arg["name"]] = create_component(arg)

                with gr.Accordion(i18n("lbl_additional_args"), open=False):
                    ui_components["additional_args"] = gr.Textbox(label=i18n("lbl_additional_args"), placeholder=i18n("ph_additional_args"))

                start_training_btn = gr.Button(i18n("btn_start_training"), variant="primary")
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
                        ui_components["input_lora_path"] = gr.Textbox(label=i18n("lbl_input_lora"), placeholder="Path to trained .safetensors file", max_lines=1)
                        ui_components["output_comfy_lora_path"] = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder="Path to save converted model", max_lines=1)
                    
                    with gr.Row():
                        set_post_defaults_btn = gr.Button(i18n("btn_set_paths"), scale=0)
                        convert_btn = gr.Button(i18n("btn_convert"), variant="secondary")
                    
                    conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)

        def update_model_visibility(arch):
            if not arch:
                return [gr.update(visible=False)] * len(model_arg_groups)
            
            updates = []
            for key, group in model_arg_groups.items():
                visible = key in arch
                if key == "Wan" and "Wan" in arch: visible = True
                if key == "Wan 2.2" and "Wan 2.2" not in arch: visible = False 
                
                updates.append(gr.update(visible=visible))
            return updates

        model_arch.change(
            fn=update_model_visibility,
            inputs=[model_arch],
            outputs=list(model_arg_groups.values())
        )

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

        all_component_keys = list(ui_components.keys())
        all_components_list = [ui_components[k] for k in all_component_keys]

        def init_project(path):
            if not path:
                return ["Please enter a project directory path."] + [gr.update()] * (len(all_components_list) + 2)
            try:
                os.makedirs(os.path.join(path, "training"), exist_ok=True)
                settings = load_project_settings(path)

                comp_values = []
                for k in all_component_keys:
                    arg_def = arg_defs_map.get(k)
                    default_val = arg_def.get("default") if arg_def else None
                    
                    if k == "project_dir":
                        val = path
                    elif k in settings:
                        val = settings[k]
                    elif default_val is not None:
                        val = default_val
                    else:
                        val = gr.update()
                    
                    comp_values.append(val)

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
                
                return [f"Project initialized at {path}.", preview_content, loaded_state] + comp_values
            except Exception as e:
                return [f"Error initializing project: {str(e)}"] + [gr.update()] * (len(all_components_list) + 2)

        def set_training_defaults(project_path, comfy_models_dir, model_arch, vram_val):
            defaults = config_manager.get_training_defaults(model_arch, vram_val, comfy_models_dir)
            updates = []

            for k in all_component_keys:
                if k in defaults:
                    updates.append(defaults[k])
                elif k in arg_defs_map and "default" in arg_defs_map[k]:
                    updates.append(arg_defs_map[k]["default"])
                else:
                    updates.append(gr.update())

            return updates

        def generate_command_generic(*args):
            values = dict(zip(all_component_keys, args))
            
            project_path = values.get("project_dir")
            if not project_path: return "Project directory not set."
            
            dataset_config = os.path.join(project_path, "dataset_config.toml")
            output_dir = os.path.join(project_path, "models")
            logging_dir = os.path.join(project_path, "logs")
            
            model = values.get("model_arch")
            if not model: return "Model not selected."

            model_conf = arg_loader.get_model_config(model)
            script_name = model_conf.get("script")
            if not script_name: return "Unknown model script."
            
            cmd = ["accelerate", "launch", os.path.join("src", "musubi_tuner", script_name)]

            for name, val in values.items():
                arg_def = arg_defs_map.get(name)
                if not arg_def: continue

                if arg_def.get("skip_if_default") and val == arg_def.get("default"):
                    continue

                cli_flag = arg_loader.get_cli_flag(name)
                if "cli_arg" in arg_def: cli_flag = arg_def["cli_arg"]
                
                if arg_def["type"] == "checkbox":
                    if val: cmd.append(cli_flag)
                elif val is not None and val != "":
                    cmd.extend([cli_flag, str(val)])

            cmd.extend([
                "--dataset_config", dataset_config,
                "--output_dir", output_dir,
                "--logging_dir", logging_dir,
                "--log_with", "tensorboard"
            ])

            task = model_conf.get("task")
            if task:
                cmd.extend(["--task", task])
                
            return cmd

        def generate_cmd_preview(*args):
            cmd = generate_command_generic(*args)
            if isinstance(cmd, str): return cmd
            return subprocess.list2cmdline([str(x) for x in cmd])

        def start_training(*args):
            cmd = generate_command_generic(*args)
            if isinstance(cmd, str): return f"Error: {cmd}"

            values = dict(zip(all_component_keys, args))
            proj = values.get("project_dir")
            if proj: save_project_settings(proj, **values)

            cmd_str = subprocess.list2cmdline([str(x) for x in cmd])
            try:
                final_cmd = f"{cmd_str} & echo. & echo Training finished. & pause >nul"
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                subprocess.Popen(["cmd", "/c", final_cmd], creationflags=flags, shell=False)
                return f"Training started!\nCommand: {cmd_str}"
            except Exception as e:
                return f"Error: {e}"

        def cache_latents(project_path, vae, te1, te2, model, comfy, w, h, batch):
            if not project_path: yield "Error: No project."; return
            if not vae: yield "Error: No VAE."; return
            config = os.path.join(project_path, "dataset_config.toml")
            script = "zimage_cache_latents.py" 
            if "Qwen" in model: script = "qwen_image_cache_latents.py"
            cmd = f"{sys.executable} src/musubi_tuner/{script} --dataset_config {config} --vae {vae}"
            yield from run_command(cmd)

        def cache_text_encoder(project_path, te1, te2, vae, model, comfy, w, h, batch):
            if not project_path: yield "Error: No project."; return
            if not te1: yield "Error: No TE1."; return
            config = os.path.join(project_path, "dataset_config.toml")
            script = "zimage_cache_text_encoder_outputs.py"
            if "Qwen" in model: script = "qwen_image_cache_text_encoder_outputs.py"
            cmd = f"{sys.executable} src/musubi_tuner/{script} --dataset_config {config} --text_encoder {te1} --batch_size 1"
            yield from run_command(cmd)
            
        def convert_lora(project_path, input_path, output_path):
             if not project_path or not input_path or not output_path:
                 yield "Error: Check paths."
                 return
             script = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
             cmd = f"{sys.executable} {script} {input_path} {output_path}"
             yield from run_command(cmd)

        def generate_config_wrapper(ds_state, proj, w, h, reps, batch, cap, bucket, no_up, model, vram, comfy, vae, te1, te2):
            return generate_config(ds_state, proj, w, h, reps, batch, cap, bucket, no_up, model, vram, comfy, vae, te1, te2)

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

        init_btn.click(
            fn=init_project,
            inputs=[ui_components["project_dir"]],
            outputs=[project_status, toml_preview, dataset_state] + all_components_list
        )

        for comp in all_components_list:
            if hasattr(comp, "change"):
                comp.change(fn=generate_cmd_preview, inputs=all_components_list, outputs=[cmd_preview])

        start_training_btn.click(fn=start_training, inputs=all_components_list, outputs=[training_status])
        copy_cmd_btn.click(fn=None, inputs=[cmd_preview], outputs=None, js="(x) => { navigator.clipboard.writeText(x); alert('Command copied to clipboard!'); }")
        add_img_ds_btn.click(fn=add_image_dataset, inputs=[dataset_state, ui_components["resolution_w"], ui_components["resolution_h"], ui_components["num_repeats"], ui_components["batch_size"], ui_components["caption_extension"], ui_components["enable_bucket"], ui_components["bucket_no_upscale"]], outputs=[dataset_state])
        add_vid_ds_btn.click(fn=add_video_dataset, inputs=[dataset_state, ui_components["resolution_w"], ui_components["resolution_h"], ui_components["num_repeats"], ui_components["batch_size"], ui_components["caption_extension"], ui_components["enable_bucket"], ui_components["bucket_no_upscale"]], outputs=[dataset_state])
        gen_toml_btn.click(fn=generate_config_wrapper, inputs=[dataset_state, ui_components["project_dir"], ui_components["resolution_w"], ui_components["resolution_h"], ui_components["num_repeats"], ui_components["batch_size"], ui_components["caption_extension"], ui_components["enable_bucket"], ui_components["bucket_no_upscale"], ui_components["model_arch"], ui_components["vram_size"], ui_components["comfy_models_dir"], ui_components["vae_path"], ui_components["text_encoder1_path"], ui_components["text_encoder2_path"]], outputs=[dataset_status, toml_preview])
        validate_models_btn.click(fn=validate_models_dir, inputs=[ui_components["comfy_models_dir"]], outputs=[models_status])
        rec_params_btn.click(fn=set_recommended_settings, inputs=[ui_components["project_dir"], ui_components["model_arch"], ui_components["vram_size"]], outputs=[ui_components["resolution_w"], ui_components["resolution_h"], ui_components["batch_size"]])
        set_rec_settings_btn.click(fn=set_recommended_settings, inputs=[ui_components["project_dir"], ui_components["model_arch"], ui_components["vram_size"]], outputs=[ui_components["resolution_w"], ui_components["resolution_h"], ui_components["batch_size"]])
        set_paths_btn.click(fn=set_preprocessing_defaults, inputs=[ui_components["project_dir"], ui_components["comfy_models_dir"], ui_components["model_arch"]], outputs=[ui_components["vae_path"], ui_components["text_encoder1_path"], ui_components["text_encoder2_path"]])
        set_training_defaults_btn.click(fn=set_training_defaults, inputs=[ui_components["project_dir"], ui_components["comfy_models_dir"], ui_components["model_arch"], ui_components["vram_size"]], outputs=all_components_list)
        cache_latents_btn.click(fn=cache_latents, inputs=[ui_components["project_dir"], ui_components["vae_path"], ui_components["text_encoder1_path"], ui_components["text_encoder2_path"], ui_components["model_arch"], ui_components["comfy_models_dir"], ui_components["resolution_w"], ui_components["resolution_h"], ui_components["batch_size"]], outputs=[training_log])
        cache_text_btn.click(fn=cache_text_encoder, inputs=[ui_components["project_dir"], ui_components["text_encoder1_path"], ui_components["text_encoder2_path"], ui_components["vae_path"], ui_components["model_arch"], ui_components["comfy_models_dir"], ui_components["resolution_w"], ui_components["resolution_h"], ui_components["batch_size"]], outputs=[training_log])
        convert_btn.click(fn=convert_lora, inputs=[ui_components["project_dir"], ui_components["input_lora_path"], ui_components["output_comfy_lora_path"]], outputs=[conversion_log])

    return demo

if __name__ == "__main__":
    with gr.Blocks(title="Musubi Tuner GUI", css=load_custom_css()) as demo:
        construct_ui()
    demo.launch()