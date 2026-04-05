#!/usr/bin/env python3
"""Keep only rows with a specific mode value from a CSV file."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only rows where the selected column exactly matches the given mode."
    )
    parser.add_argument("csv_path", type=Path, help="Path to input CSV (e.g. combined.ALL.csv).")
    parser.add_argument("mode", help="Mode value to keep (exact match).")
    parser.add_argument(
        "--column",
        default="mode",
        help="Column name to filter on (default: mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to '<input>.<mode>.only.csv'.",
    )
    return parser.parse_args()


def sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return cleaned[:60] if cleaned else "mode"


def default_output_path(csv_path: Path, mode: str) -> Path:
    suffix = sanitize(mode)
    if csv_path.suffix:
        return csv_path.with_name(f"{csv_path.stem}.{suffix}.only{csv_path.suffix}")
    return csv_path.with_name(f"{csv_path.name}.{suffix}.only")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if not args.csv_path.is_file():
        raise SystemExit(f"CSV not found: {args.csv_path}")

    with args.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV appears empty: {args.csv_path}")
        if args.column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise SystemExit(
                f"Column '{args.column}' not found in {args.csv_path}. Available columns: {available}"
            )
        rows = list(reader)

    kept_rows = [row for row in rows if row.get(args.column) == args.mode]
    dropped = len(rows) - len(kept_rows)

    output_path = args.output if args.output is not None else default_output_path(args.csv_path, args.mode)
    write_csv(output_path, reader.fieldnames, kept_rows)

    print(f"Input rows: {len(rows)}")
    print(f"Kept rows where {args.column} == '{args.mode}': {len(kept_rows)}")
    print(f"Dropped rows: {dropped}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
