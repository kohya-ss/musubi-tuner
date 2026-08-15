import argparse

import torch

from musubi_tuner.flux import flux_utils
from musubi_tuner.flux_kontext_train_network import FluxKontextNetworkTrainer, flux_kontext_setup_parser
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.training.full_finetune import FullFineTuningTrainerMixin, add_full_finetune_args


class FluxKontextTrainer(FullFineTuningTrainerMixin, FluxKontextNetworkTrainer):
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
        return flux_utils.load_flow_model(
            ckpt_path=dit_path,
            dtype=trainable_dtype,
            device=loading_device,
            disable_mmap=True,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            fp8_scaled=False,
        )


def main():
    parser = setup_parser_common()
    parser = flux_kontext_setup_parser(parser)
    parser = add_full_finetune_args(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = FluxKontextTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
