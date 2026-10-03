"""Convert a DLSS-NR 310.8.0 model directory into canonical FP32 tensors."""

from __future__ import annotations

import argparse

from musubi_tuner.dlssnr.checkpoint import assert_auxiliary_fingerprint, assert_source_fingerprint, load_source, measure_fingerprint
from musubi_tuner.dlssnr.convert import convert_source
from musubi_tuner.dlssnr.profiles import PROFILE_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DLSS-NR 310.8.0 packed stages into a canonical checkpoint.")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--profile", default=PROFILE_ID)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--verify_roundtrip", action="store_true")
    args = parser.parse_args()
    if args.profile != PROFILE_ID:
        raise SystemExit(f"unsupported profile {args.profile}; this build knows {PROFILE_ID}")

    source = load_source(args.source_dir)
    assert_source_fingerprint(measure_fingerprint(source))
    assert_auxiliary_fingerprint(source)
    report = convert_source(source, args.output_dir, verify=args.verify_roundtrip)
    print(f"wrote {args.output_dir} roundtrip={report['roundtrip']} parameters={report['logical_parameters']}")


if __name__ == "__main__":
    main()
