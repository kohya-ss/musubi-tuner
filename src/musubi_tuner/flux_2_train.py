import argparse

import torch

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.training.full_finetune import FullFineTuningTrainerMixin, add_full_finetune_args


class Flux2Trainer(FullFineTuningTrainerMixin, Flux2NetworkTrainer):
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
        return flux2_utils.load_flow_model(
            device=accelerator.device,
            model_version_info=self.model_version_info,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=trainable_dtype,
            fp8_scaled=False,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )

    def full_finetune_metadata(self, args: argparse.Namespace) -> dict[str, str]:
        return {"ss_flux_2_model_version": str(args.model_version)}


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = add_full_finetune_args(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    trainer = Flux2Trainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
