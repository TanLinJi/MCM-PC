#!/usr/bin/env python
"""Summarize Figure 1(a) reproduction logs into a single average number.

Usage:
    python scripts/repro_fig1a_summarize.py <log_dir>

Where <log_dir> contains *.log files, each ending with a line of the form:
    ---- ***Final*** Zero-shot test accuracy: 71.47. ----
    or
    ---- ***Final*** TDA's test accuracy: 76.59. ----

The script greps the last "Final ... :" number from every .log file and
prints both per-file results and the arithmetic mean.

Nested directories (clean/, zs_c/, tta_c/) are also handled if present;
each subdir gets its own average.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAT = re.compile(r"\*\*\*Final\*\*\*[^:]*:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_one(logfile: Path) -> float | None:
    try:
        text = logfile.read_text(errors="ignore")
    except Exception:
        return None
    matches = PAT.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def summarize_dir(d: Path, label: str = "") -> None:
    logs = sorted(d.glob("*.log"))
    if not logs:
        return
    rows: list[tuple[str, float | None]] = [
        (lf.stem, parse_one(lf)) for lf in logs
    ]

    print(f"\n=== {label or d.name}  ({len(rows)} runs) ===")
    bad = []
    vals = []
    for name, v in rows:
        if v is None:
            bad.append(name)
            print(f"  {name:32s}  FAIL (no Final line)")
        else:
            vals.append(v)
            print(f"  {name:32s}  {v:6.2f}")

    if vals:
        avg = sum(vals) / len(vals)
        print(f"\n  >>> mean over {len(vals)} runs: {avg:.2f}")
    if bad:
        print(f"  !!! {len(bad)} runs failed to parse: {bad}")


def main(root: Path) -> None:
    print(f"Log root: {root.resolve()}")

    # If subdirs exist (clean/, zs_c/, tta_c/), summarize each separately.
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        for sd in sorted(subdirs):
            summarize_dir(sd, label=sd.name)
    else:
        summarize_dir(root, label=root.name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
