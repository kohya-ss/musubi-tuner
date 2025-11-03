import importlib
import logging
import os
import built_in_extensions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StepData:
    """Class for storing data during the sampling process"""

    def __init__(self, args, context, step_index, timestep, latents, latent_model_input):
        # basic parameters
        self.args = args
        self.context = context
        self.step_index = step_index
        self.timestep = timestep
        self.latents = latents
        self.latent_model_input = latent_model_input

        # outputs from the model
        self.noise_pred = None
        self.noise_pred_cond = None
        self.noise_pred_uncond = None


class Context:
    """Class for storing context during the sampling process"""

    def __init__(self, model, scheduler, tokenizer, vae, text_encoders):
        self.model = model
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.vae = vae
        self.text_encoders = text_encoders


class Extension:
    """Base class for extensions"""

    def __init__(self, architecture: str, config=None):
        """
        initialize the extension with the given configuration.
        Args:
            config: configuration dictionary for the extension.
        """
        self.architecture = architecture
        self.config = config or {}
        self.enabled = True

    def on_init(self, args, context):
        """called when the extension is initialized"""
        return args, context

    def on_model_load_pre(self, args, context):
        """called before the model is loaded"""
        return args, context

    def on_model_load_post(self, args, context):
        """called after the model is loaded"""
        return args, context

    def on_text_encode_pre(self, args, context):
        """called before text encoding"""
        return args, context

    def on_text_encode_post(self, args, context, embeddings):
        """called after text encoding"""
        return args, context, embeddings

    def on_sampling_pre(self, args, context):
        """called before sampling starts"""
        return args, context

    def on_sampling_step(self, args, context, step_data):
        """called during each sampling step"""
        return args, context, step_data

    def on_sampling_post(self, args, context, latents):
        """called after sampling is complete"""
        return args, context, latents

    def on_decode_pre(self, args, context, latents):
        """called before decoding"""
        return args, context, latents

    def on_decode_post(self, args, context, images):
        """called after decoding"""
        return args, context, images

    def on_save_pre(self, args, context, output):
        """called before saving"""
        return args, context, output

    def on_save_post(self, args, context, output_path):
        """called after saving"""
        return args, context, output_path

    def on_cleanup(self, args, context):
        """called during cleanup"""
        return args, context


class ExtensionManager:
    """Class for managing extensions"""

    def __init__(self):
        self.extensions: list[Extension] = []
        self.extension_configs = {}

    def load_extension(self, extension_path, config=None):
        """Load an extension from a given path"""
        module_name = os.path.basename(extension_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, extension_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # search for the first class that is a subclass of Extension
        extension_class = None
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, Extension) and obj != Extension:
                extension_class = obj
                break

        if extension_class is None:
            raise ValueError(f"No valid Extension class found in {extension_path}")

        extension = extension_class(config)
        self.extensions.append(extension)
        return extension

    def load_extensions_from_config(self, config_path):
        """
        load extensions from a configuration file

        sample config file:
        ```toml
        [extensions]
        [extensions.extension_name]
        path = "path/to/extension.py"
        param1 = "value1"
        param2 = "value2"
        ```

        For built-in extensions, the path can be omitted, and the extension name will be used to load the extension.
        For example, to load the "example_extension" extension, you can use:
        ```toml
        [extensions.example_extension]
        ```
        """
        import toml

        with open(config_path, "r") as f:
            config = toml.load(f)

        extensions_config = config.get("extensions", {})
        for ext_name, ext_config in extensions_config.items():
            self.extension_configs[ext_name] = ext_config
            path = ext_config.get("path", None)
            if path:
                self.load_extension(path, ext_config)
            else:
                # load built-in extension
                extension = built_in_extensions.load_builtin_extensions(ext_name, ext_config)
                if extension is None:
                    raise ValueError(f"Failed to load built-in extension {ext_name}")
                self.extensions.append(extension)

    def call_extensions(self, method_name, *args, **kwargs):
        """calls the specified method on all loaded extensions"""
        results = list(args)

        for extension in self.extensions:
            if not extension.enabled:
                continue

            method = getattr(extension, method_name, None)
            if method is None:
                continue

            try:
                extension_results = method(*results, **kwargs)
                assert len(extension_results) == len(
                    results
                ), f"Extension {extension.__class__.__name__} returned unexpected number of results for {method_name}"

                # results = list(extension_results) if isinstance(extension_results, tuple) else extension_results
                if isinstance(extension_results, tuple):
                    results = list(extension_results)
                elif isinstance(extension_results, list):
                    results = extension_results
                else:
                    results = [extension_results]

            except Exception as e:
                logger.error(f"Extension {extension.__class__.__name__} failed to call {method_name}: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

        return results

    def call_sampling_step_extensions(self, args, context, step_data):
        return self.call_extensions("on_sampling_step", args, context, step_data)
        # for extension in self.extensions:
        #     if not extension.enabled:
        #         continue
        #     try:
        #         step_data = extension.on_sampling_step(step_data)
        #     except Exception as e:
        #         logger.error(f"拡張機能 {extension.__class__.__name__} の on_sampling_step 呼び出しに失敗: {str(e)}")
        #         import traceback
        #         logger.error(traceback.format_exc())
        # return step_data
