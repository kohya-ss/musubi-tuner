"""Inspect a DLSS-NR 310.8.0 model directory and write a fingerprint report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from musubi_tuner.dlssnr.checkpoint import (
    assert_auxiliary_fingerprint,
    assert_source_fingerprint,
    load_source,
    logical_parameter_count,
    measure_fingerprint,
)
from musubi_tuner.dlssnr.profiles import PROFILE_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a DLSS-NR 310.8.0 weight directory against the fixed profile.")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--profile", default=PROFILE_ID)
    parser.add_argument("--report_dir", required=True)
    args = parser.parse_args()
    if args.profile != PROFILE_ID:
        raise SystemExit(f"unsupported profile {args.profile}; this build knows {PROFILE_ID}")

    source = load_source(args.source_dir)
    fingerprint = measure_fingerprint(source)
    assert_source_fingerprint(fingerprint)
    assert_auxiliary_fingerprint(source)
    report = {
        "profile": PROFILE_ID,
        "payload_bytes": sum(len(blob) for blob in source.stages.values()),
        "records": len(source.records),
        "logical_parameters": logical_parameter_count(source.records),
        "stage_sha256": source.stage_sha256,
        "fingerprint": fingerprint,
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "inspect_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{PROFILE_ID}: {report['records']} records, {report['payload_bytes']} bytes, "
        f"{report['logical_parameters']} logical parameters"
    )


if __name__ == "__main__":
    main()
