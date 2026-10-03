"""DLSS-NR LoRA training with dataset TOML and command-line hyperparameters."""

from __future__ import annotations

import logging

from musubi_tuner.training.dlssnr_parser import setup_parser as setup_nr_parser
from musubi_tuner.training.dlssnr_trainer import train_lora_from_args


def setup_parser():
    return setup_nr_parser(lora=True)


def main() -> None:
    args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    train_lora_from_args(args)


if __name__ == "__main__":
    main()
