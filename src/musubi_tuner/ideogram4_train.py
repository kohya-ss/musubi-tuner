import argparse

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_quantized_loading import FP8_SCALE_SUFFIX, is_bnb4bit_state_dict
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer, ideogram4_setup_parser
from musubi_tuner.training.full_finetune import FullFineTuningTrainerMixin, add_full_finetune_args
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


class Ideogram4Trainer(FullFineTuningTrainerMixin, Ideogram4NetworkTrainer):
    def validate_full_finetune_model_args(self, args: argparse.Namespace) -> None:
        if args.use_unconditional_dit_for_lora_sampling:
            raise ValueError(
                "--use_unconditional_dit_for_lora_sampling is LoRA-only; full finetuning uses "
                "--unconditional_dit automatically for sampling"
            )

        ideogram4_utils.validate_local_safetensors(
            args.dit,
            expected_model_type=ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
        )
        with MemoryEfficientSafeOpen(
            args.dit,
            disable_numpy_memmap=args.disable_numpy_memmap,
        ) as checkpoint:
            keys = checkpoint.keys()
            tensor_dtypes = {value.get("dtype") for key, value in checkpoint.header.items() if key != "__metadata__"}

        if is_bnb4bit_state_dict(keys):
            raise ValueError("Ideogram 4 full finetuning does not support bnb 4-bit conditional DiT checkpoints")
        if any(key.endswith(FP8_SCALE_SUFFIX) for key in keys) or tensor_dtypes.intersection({"F8_E4M3", "F8_E5M2"}):
            raise ValueError("Ideogram 4 full finetuning does not support prequantized FP8 conditional DiT checkpoints")
        unsupported_dtypes = tensor_dtypes.difference({"F32", "F16", "BF16"})
        if unsupported_dtypes:
            found = ", ".join(sorted(str(dtype) for dtype in unsupported_dtypes))
            raise ValueError(f"Ideogram 4 full-finetune checkpoint tensor dtype must be F32, F16, or BF16; found {found}")

    def use_unconditional_dit_for_sampling(self, args: argparse.Namespace) -> bool:
        return bool(args.unconditional_dit)

    def full_finetune_metadata(self, args: argparse.Namespace) -> dict[str, str]:
        del args
        return {"model_type": ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE}


def main():
    parser = setup_parser_common()
    parser = ideogram4_setup_parser(parser)
    parser = add_full_finetune_args(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Ideogram4Trainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
