from extensions import Extension


def load_builtin_extensions(ext_name: str, ext_config: dict):
    """Load built-in extensions."""

    built_in_extensions = {
        "no_op_extension": Extension,
        "cfg_skip": CfgSkipExtension,
    }
    if ext_name in built_in_extensions:
        extension_class = built_in_extensions[ext_name]
        return extension_class(ext_config)
    else:
        raise ValueError(f"Extension {ext_name} not found in built-in extensions.")


class CfgSkipExtension(Extension):
    """A built-in extension that skips the configuration."""

    def __init__(self, architecture: str, config: dict):
        super().__init__(architecture, config)

    def on_sampling_step(self, args, context, step_data):
        return super().on_sampling_step(args, context, step_data)
