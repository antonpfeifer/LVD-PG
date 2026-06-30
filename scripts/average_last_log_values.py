#!/usr/bin/env python3
"""Average trn and val values from the last line of each log file.

By default this reads logs from:
  exps/progressive_growing/temp/temp_wikitext/logs/id_poswise_cat_l2_id20_init2_final4

Only the final non-empty line of each *.log file is used.
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
    rf"\btrn:\s*(?P<trn>{NUMBER_RE})\s+val:\s*(?P<val>{NUMBER_RE})\b",
    re.IGNORECASE,
)


def last_non_empty_line(path: Path) -> str | None:
    """Return the last non-empty line in path, or None if the file is empty."""
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        buffer = bytearray()

        while pos > 0:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == b"\n":
                line = bytes(reversed(buffer)).decode("utf-8", errors="replace").strip()
                if line:
                    return line
                buffer.clear()
            else:
                buffer.append(char[0])

        if buffer:
            return bytes(reversed(buffer)).decode("utf-8", errors="replace").strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average trn and val from the last line of each log file."
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
        line = last_non_empty_line(path)
        if line is None:
            parse_skipped.append((path, "empty file"))
            continue

        match = TRN_VAL_RE.search(line)
        if not match:
            parse_skipped.append((path, f"could not parse last line: {line!r}"))
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
    if trn_values:
        print(f"avg trn: {sum(trn_values) / len(trn_values):.6f} ({len(trn_values)} values)")
    else:
        print("avg trn: unavailable (0 finite values)")

    if val_values:
        print(f"avg val: {sum(val_values) / len(val_values):.6f} ({len(val_values)} values)")
    else:
        print("avg val: unavailable (0 finite values)")

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
