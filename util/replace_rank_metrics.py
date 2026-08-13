#!/usr/bin/env python3
"""Replace KR and GR in one results CSV using values from another CSV.

The target CSV supplies the rows and every non-rank value. The ranks CSV
supplies only the replacement KR and GR values. Rows are matched by experiment
identity rather than by file order.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import shutil
import tempfile


CORE_KEY_COLUMNS = (
    "mode",
    "shuffle_id",
    "rho_target",
    "leak",
    "input_scale",
    "neuron_bias",
)
OPTIONAL_KEY_COLUMNS = ("normalization", "seed", "src")
NUMERIC_KEY_COLUMNS = {
    "shuffle_id",
    "rho_target",
    "leak",
    "input_scale",
    "neuron_bias",
    "seed",
}
RANK_COLUMNS = ("KR", "GR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy KR and GR from a ranks CSV into matching rows of a target CSV, "
            "while preserving MC, IPC, row order, and every other target column."
        )
    )
    parser.add_argument(
        "target_csv",
        type=Path,
        help="Existing CSV whose KR/GR values should be replaced.",
    )
    parser.add_argument(
        "ranks_csv",
        type=Path,
        help="CSV containing the corrected KR/GR values.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="Output path. The default is '<target>.rank_updated.csv'.",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Atomically replace the target CSV and save its original contents as "
            "'<target>.pre_rank_update.bak'."
        ),
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        help=(
            "Columns used to match rows. By default, the script uses mode, shuffle_id, "
            "rho_target, leak, input_scale, neuron_bias, plus normalization, seed, and "
            "src when present in both files."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Update matching rows even if the two CSVs do not contain identical row keys.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow an existing --output/default output file to be overwritten.",
    )
    return parser.parse_args()


def default_output_path(target: Path) -> Path:
    if target.suffix:
        return target.with_name(f"{target.stem}.rank_updated{target.suffix}")
    return target.with_name(f"{target.name}.rank_updated.csv")


def backup_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.pre_rank_update.bak")


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
    if not header:
        raise ValueError(f"CSV has no header: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"CSV has duplicate column names: {path}")
    return header


def choose_key_columns(
    target_header: list[str],
    ranks_header: list[str],
    requested: list[str] | None,
) -> tuple[str, ...]:
    common = set(target_header) & set(ranks_header)
    if requested:
        missing = [column for column in requested if column not in common]
        if missing:
            raise ValueError(
                "Requested key columns missing from one or both CSVs: " + ", ".join(missing)
            )
        return tuple(requested)

    missing_core = [column for column in CORE_KEY_COLUMNS if column not in common]
    if missing_core:
        raise ValueError(
            "Cannot determine the default row identity; missing columns: "
            + ", ".join(missing_core)
            + ". Pass --key-columns explicitly if these files use another schema."
        )
    return CORE_KEY_COLUMNS + tuple(
        column for column in OPTIONAL_KEY_COLUMNS if column in common
    )


def canonical_value(column: str, value: str | None) -> str:
    text = "" if value is None else value.strip()
    if column not in NUMERIC_KEY_COLUMNS or not text:
        return text
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric key value {text!r} in column {column!r}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite key value {text!r} in column {column!r}")
    if number == 0:
        return "0"
    return str(number.normalize())


def row_key(row: dict[str, str], key_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonical_value(column, row.get(column)) for column in key_columns)


def validate_rank(value: str | None, column: str, row_number: int) -> str:
    text = "" if value is None else value.strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} value {text!r} in ranks CSV row {row_number}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {column} value in ranks CSV row {row_number}")
    return text


def load_replacement_ranks(
    path: Path,
    key_columns: tuple[str, ...],
) -> dict[tuple[str, ...], tuple[str, str]]:
    replacements: dict[tuple[str, ...], tuple[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            key = row_key(row, key_columns)
            if key in replacements:
                raise ValueError(
                    f"Duplicate row key in ranks CSV at row {row_number}: {key}"
                )
            replacements[key] = (
                validate_rank(row.get("KR"), "KR", row_number),
                validate_rank(row.get("GR"), "GR", row_number),
            )
    return replacements


def format_examples(keys: list[tuple[str, ...]], limit: int = 3) -> str:
    return "; ".join(repr(key) for key in keys[:limit])


def replace_rank_metrics(
    target: Path,
    ranks: Path,
    output: Path,
    key_columns: tuple[str, ...],
    allow_partial: bool,
    backup: Path | None = None,
) -> tuple[int, int, int]:
    replacements = load_replacement_ranks(ranks, key_columns)
    output.parent.mkdir(parents=True, exist_ok=True)

    matched_keys: set[tuple[str, ...]] = set()
    target_keys: set[tuple[str, ...]] = set()
    unmatched_target: list[tuple[str, ...]] = []
    updated_rows = 0
    total_rows = 0

    with tempfile.NamedTemporaryFile(
        "w",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with target.open("r", newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                assert reader.fieldnames is not None
                writer = csv.DictWriter(temporary, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row_number, row in enumerate(reader, start=2):
                    total_rows += 1
                    key = row_key(row, key_columns)
                    if key in target_keys:
                        raise ValueError(
                            f"Duplicate row key in target CSV at row {row_number}: {key}"
                        )
                    target_keys.add(key)
                    replacement = replacements.get(key)
                    if replacement is None:
                        unmatched_target.append(key)
                    else:
                        row["KR"], row["GR"] = replacement
                        matched_keys.add(key)
                        updated_rows += 1
                    writer.writerow(row)

            unused_source = [key for key in replacements if key not in matched_keys]
            if not allow_partial and (unmatched_target or unused_source):
                messages = []
                if unmatched_target:
                    messages.append(
                        f"{len(unmatched_target)} target rows have no replacement; examples: "
                        f"{format_examples(unmatched_target)}"
                    )
                if unused_source:
                    messages.append(
                        f"{len(unused_source)} ranks rows have no target; examples: "
                        f"{format_examples(unused_source)}"
                    )
                raise ValueError("Row sets do not match. " + " ".join(messages))

            temporary.flush()
            if backup is not None:
                shutil.copy2(target, backup)
            temporary_path.replace(output)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    return total_rows, updated_rows, len(replacements) - len(matched_keys)


def main() -> int:
    args = parse_args()
    for path in (args.target_csv, args.ranks_csv):
        if not path.is_file():
            raise SystemExit(f"CSV not found: {path}")

    try:
        target_header = read_header(args.target_csv)
        ranks_header = read_header(args.ranks_csv)
        for path, header in (
            (args.target_csv, target_header),
            (args.ranks_csv, ranks_header),
        ):
            missing = [column for column in RANK_COLUMNS if column not in header]
            if missing:
                raise ValueError(f"Missing {', '.join(missing)} column(s) in {path}")

        key_columns = choose_key_columns(
            target_header,
            ranks_header,
            args.key_columns,
        )
        output = args.target_csv if args.in_place else (
            args.output or default_output_path(args.target_csv)
        )
        if not args.in_place and output.exists() and not args.force:
            raise ValueError(f"Output already exists: {output}. Pass --force to replace it.")

        if args.in_place:
            backup = backup_path(args.target_csv)
            if backup.exists() and not args.force:
                raise ValueError(
                    f"Backup already exists: {backup}. Move it or pass --force to replace it."
                )
        else:
            backup = None

        total, updated, unused = replace_rank_metrics(
            args.target_csv,
            args.ranks_csv,
            output,
            key_columns,
            args.allow_partial,
            backup,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Matched on: {', '.join(key_columns)}")
    print(f"Target rows: {total}")
    print(f"KR/GR rows updated: {updated}")
    if args.allow_partial:
        print(f"Target rows left unchanged: {total - updated}")
        print(f"Unused rows in ranks CSV: {unused}")
    if backup is not None:
        print(f"Backup: {backup}")
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
