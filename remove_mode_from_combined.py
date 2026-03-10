#!/usr/bin/env python3
"""Remove rows with a specific mode value from a combined CSV file."""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rows where the mode column matches the given mode."
    )
    parser.add_argument("csv_path", type=Path, help="Path to input CSV (e.g. combined.ALL.csv).")
    parser.add_argument("mode", help="Mode value to remove (exact match).")
    parser.add_argument(
        "--column",
        default="mode",
        help="Column name to filter on (default: mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to '<input>.filtered.csv'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input CSV instead of writing a new file.",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    if csv_path.suffix:
        return csv_path.with_name(f"{csv_path.stem}.filtered{csv_path.suffix}")
    return csv_path.with_name(f"{csv_path.name}.filtered")


def sanitize_for_temp(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:40]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or --output, not both.")
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

    kept_rows = [row for row in rows if row.get(args.column) != args.mode]
    removed = len(rows) - len(kept_rows)

    if args.in_place:
        suffix = sanitize_for_temp(args.mode)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=args.csv_path.parent,
            prefix=f".tmp_remove_{suffix}_",
            newline="",
            encoding="utf-8",
        ) as tmp:
            tmp_path = Path(tmp.name)
        write_csv(tmp_path, reader.fieldnames, kept_rows)
        tmp_path.replace(args.csv_path)
        output_path = args.csv_path
    else:
        output_path = args.output if args.output is not None else default_output_path(args.csv_path)
        write_csv(output_path, reader.fieldnames, kept_rows)

    print(f"Input rows: {len(rows)}")
    print(f"Removed rows where {args.column} == '{args.mode}': {removed}")
    print(f"Output rows: {len(kept_rows)}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
