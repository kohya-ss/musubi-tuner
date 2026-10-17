"""Write HT/CC canonical weights or CC packed records into a copy of an original DLL."""

from __future__ import annotations

import argparse
import json
import logging

from musubi_tuner.dlssnr.dll_models import pack_selected_model


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack selected DLSS-NR HT or CC weights into a new DLL; native loading is not certified."
    )
    parser.add_argument("--template_dll", required=True, help="DLL with the original audited selected resource; never modified")
    parser.add_argument(
        "--model_dir",
        "--record_dir",
        "--input",
        required=True,
        help="HT/CC canonical checkpoint, CC record directory, or multi-model output with --weights",
    )
    parser.add_argument(
        "--weights", choices=("HT", "CC"), help="Select a bundle child; otherwise inferred from the model directory"
    )
    parser.add_argument("--output_dll", "--output", required=True, help="New DLL path; also writes <path>.report.json")
    parser.add_argument(
        "--mix", type=float, default=1.0, help="Canonical weights: template/trained interpolation in [0,1]; default 1=trained"
    )
    parser.add_argument(
        "--strength",
        "--multiplier",
        type=float,
        default=1.0,
        help="Canonical weights: scale ALL numeric weights after mixing (default 1); not visual effect strength.",
    )
    parser.add_argument(
        "--merge_lora", "--lora_weight", help="DLSS-NR adapter.safetensors to merge into a canonical --model_dir before packing"
    )
    parser.add_argument(
        "--lora_multiplier", type=float, default=1.0, help="Canonical weights: scale the LoRA delta before mixing (default 1)"
    )
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = pack_selected_model(
            args.template_dll,
            args.model_dir,
            args.output_dll,
            weights=args.weights,
            mix=args.mix,
            strength=args.strength,
            merge_lora=args.merge_lora,
            lora_multiplier=args.lora_multiplier,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    keys = (
        "resource_name",
        "output_dll",
        "output_sha256",
        "whole_dll_byte_identical",
        "totals",
        "changed_records",
        "changed_bytes",
    )
    summary = {key: report[key] for key in keys if key in report}
    summary["report"] = str(args.output_dll) + ".report.json"
    print(json.dumps(summary, indent=2))
    if "totals" in report and report["totals"]["blended_changed_values"] and not report["totals"]["exported_changed_values"]:
        logging.warning("All requested changes rounded back to the template weights; the DLL is unchanged.")
    logging.warning("Native loading/image quality is unverified; no PE checksum or signature repair is performed.")


if __name__ == "__main__":
    main()
