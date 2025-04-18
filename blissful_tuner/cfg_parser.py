#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:37:25 2025

@author: blyss
"""

def parse_scheduled_cfg(schedule: str, infer_steps: int) -> List[int]:
    """
    Parse a schedule string like "1-10,20,!5,e~3" into a sorted list of steps.

    - "start-end" includes all steps in [start, end]
    - "e~n"    includes every nth step (n, 2n, ...) up to infer_steps
    - "x"      includes the single step x
    - Prefix "!" on any token to exclude those steps instead of including them.

    Raises argparse.ArgumentTypeError on malformed tokens or out-of-range steps.
    """
    included = set()
    excluded = set()

    for raw in schedule.split(","):
        token = raw.strip()
        if not token:
            continue  # skip empty tokens

        # exclusion if it starts with "!"
        if token.startswith("!"):
            target = "exclude"
            token = token[1:]
        else:
            target = "include"

        # modulus syntax: e.g. "e~3"
        if token.startswith("e~"):
            num_str = token[2:]
            try:
                n = int(num_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid modulus in '{raw}'")
            if n < 1:
                raise argparse.ArgumentTypeError(f"Modulus must be ≥ 1 in '{raw}'")

            steps = range(n, infer_steps + 1, n)

        # range syntax: e.g. "5-10"
        elif "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Malformed range '{raw}'")
            start_str, end_str = parts
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Non‑integer in range '{raw}'")
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError(f"Steps must be ≥ 1 in '{raw}'")
            if start > end:
                raise argparse.ArgumentTypeError(f"Start > end in '{raw}'")
            if end > infer_steps:
                raise argparse.ArgumentTypeError(f"End > infer_steps ({infer_steps}) in '{raw}'")

            steps = range(start, end + 1)

        # single‑step syntax: e.g. "7"
        else:
            try:
                step = int(token)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid token '{raw}'")
            if step < 1 or step > infer_steps:
                raise argparse.ArgumentTypeError(f"Step {step} out of range 1–{infer_steps} in '{raw}'")

            steps = [step]

        # apply include/exclude
        if target == "include":
            included.update(steps)
        else:
            excluded.update(steps)

    # final steps = included minus excluded, sorted
    return sorted(s for s in included if s not in excluded)