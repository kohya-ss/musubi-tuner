"""DLSS-NR supervised training. Mixed precision, checkpointing and multi-GPU are rejected."""

from __future__ import annotations

import logging

from musubi_tuner.training.dlssnr_parser import setup_parser
from musubi_tuner.training.dlssnr_trainer import train_from_args


def main() -> None:
    args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    train_from_args(args)


if __name__ == "__main__":
    main()
