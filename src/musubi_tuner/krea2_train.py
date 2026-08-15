import argparse

import torch

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.training.full_finetune import FullFineTuningTrainerMixin, add_full_finetune_args


def krea2_full_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dit_variant",
        type=str,
        choices=["raw", "turbo"],
        default="raw",
        help="variant of the full-finetune --dit checkpoint; Turbo uses the fixed mu=1.15 sampling schedule",
    )
    return parser


class Krea2Trainer(FullFineTuningTrainerMixin, Krea2NetworkTrainer):
    def validate_full_finetune_model_args(self, args: argparse.Namespace) -> None:
        if args.dit_variant not in {"raw", "turbo"}:
            raise ValueError(f"--dit_variant must be 'raw' or 'turbo', got {args.dit_variant!r}")
        temporary_turbo_options = []
        if args.turbo_dit is not None:
            temporary_turbo_options.append("--turbo_dit")
        if args.turbo_dit_cache:
            temporary_turbo_options.append("--turbo_dit_cache")
        if temporary_turbo_options:
            raise ValueError(
                "full finetuning does not support LoRA-only temporary Turbo weight swapping: "
                + ", ".join(temporary_turbo_options)
                + "; pass the Turbo checkpoint as --dit together with --dit_variant turbo"
            )

    def use_turbo_sampling_schedule(self, args: argparse.Namespace) -> bool:
        return args.dit_variant == "turbo"

    def load_full_finetune_transformer(
        self,
        accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device,
        trainable_dtype: torch.dtype,
    ):
        del accelerator, args
        return krea2_utils.load_krea2_dit(
            dit_path,
            device=loading_device,
            dtype=trainable_dtype,
            fp8_scaled=False,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )

    def full_finetune_metadata(self, args: argparse.Namespace) -> dict[str, str]:
        return {"ss_krea2_dit_variant": str(args.dit_variant)}


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = krea2_full_setup_parser(parser)
    parser = add_full_finetune_args(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2Trainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
