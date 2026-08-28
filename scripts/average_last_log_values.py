#!/usr/bin/env python3
"""Average trn and val values from the 3rd line of each log file.

By default this reads logs from:
  exps/progressive_growing/temp/temp_wikitext/logs/id_poswise_cat_l2_id20_init2_final4

Only line 3 of each *.log file is used.  If that line contains the
progressive-growing format ``trn: bpd(perplexity) val: bpd(perplexity)``, the
bpd values are averaged and perplexity is reported as ``exp(avg_bpd * log(2))``
(i.e. ``2^avg_bpd``), matching parallel_PG.jl.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

DEFAULT_LOG_DIR = Path(
    "exps/progressive_growing/temp/temp_wikitext/logs/"
    "id_poswise_cat_l2_id200_init2_final4"
)

NUMBER_RE = r"(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf(?:inity)?)"
TRN_VAL_RE = re.compile(
    rf"\btrn:\s*(?P<trn>{NUMBER_RE})"
    rf"(?:\s*\(\s*(?P<trn_perplexity>{NUMBER_RE})\s*\))?"
    rf"\s+val:\s*(?P<val>{NUMBER_RE})"
    rf"(?:\s*\(\s*(?P<val_perplexity>{NUMBER_RE})\s*\))?\b",
    re.IGNORECASE,
)


def bpd_to_perplexity(bpd: float) -> float:
    """Convert bpd to perplexity as in parallel_PG.jl: exp(bpd * log(2))."""
    return math.exp(bpd * math.log(2.0))


def third_line(path: Path) -> str | None:
    """Return exactly line 3 in path, or None if the file has fewer lines."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number == 3:
                return line.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average trn and val from the 3rd line of each log file."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing log files (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern for log files inside log_dir (default: *.log)",
    )
    args = parser.parse_args()

    if not args.log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {args.log_dir}")

    trn_values: list[float] = []
    val_values: list[float] = []
    parse_skipped: list[tuple[Path, str]] = []
    non_finite_skipped: list[tuple[Path, str]] = []
    files_seen = 0

    for path in sorted(args.log_dir.glob(args.pattern)):
        if not path.is_file():
            continue

        files_seen += 1
        line = third_line(path)
        if line is None:
            parse_skipped.append((path, "fewer than 3 lines"))
            continue

        match = TRN_VAL_RE.search(line)
        if not match:
            parse_skipped.append((path, f"could not parse 3rd line: {line!r}"))
            continue

        trn = float(match.group("trn"))
        val = float(match.group("val"))

        if math.isfinite(trn):
            trn_values.append(trn)
        else:
            non_finite_skipped.append((path, f"trn is {match.group('trn')}"))

        if math.isfinite(val):
            val_values.append(val)
        else:
            non_finite_skipped.append((path, f"val is {match.group('val')}"))

    if not trn_values and not val_values:
        raise SystemExit("No parseable finite trn or val values found.")

    print(f"files scanned: {files_seen}")
    avg_trn = sum(trn_values) / len(trn_values) if trn_values else None
    avg_val = sum(val_values) / len(val_values) if val_values else None

    if avg_trn is not None:
        print(f"avg trn: {avg_trn:.6f} ({len(trn_values)} values)")
    else:
        print("avg trn: unavailable (0 finite values)")

    if avg_val is not None:
        print(f"avg val: {avg_val:.6f} ({len(val_values)} values)")
    else:
        print("avg val: unavailable (0 finite values)")

    print("perplexity (2^avg bpd):")
    if avg_trn is not None:
        print(f"  trn: {bpd_to_perplexity(avg_trn):.6f}")
    else:
        print("  trn: unavailable")
    if avg_val is not None:
        print(f"  val: {bpd_to_perplexity(avg_val):.6f}")
    else:
        print("  val: unavailable")

    if parse_skipped:
        print(f"unparseable files: {len(parse_skipped)}")
        for path, reason in parse_skipped:
            print(f"  {path}: {reason}")

    if non_finite_skipped:
        print(f"non-finite values skipped: {len(non_finite_skipped)}")
        for path, reason in non_finite_skipped:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()

