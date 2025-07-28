#!/usr/bin/env python3
"""
print_parquet_field.py – print the values of one or more columns in a Parquet file
                                          using Hugging Face `datasets`.

Examples
--------
# Print the first 20 values in columns "text" then "id"
python print_parquet_field.py --input data/myfile.parquet --field text id --n 20

# Stream‑through and print every value in columns "id" then "meta"
python print_parquet_field.py --input bigfile.parquet --field id meta --stream
"""

import argparse
from datasets import load_dataset
from typing import List, Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print one or more columns from a Parquet file (🤗 datasets).",
    )
    p.add_argument("--input", required=True, help="Path to the Parquet file")
    # Accept one or more fields, preserving order as given by the user
    p.add_argument(
        "--field",
        required=True,
        nargs="+",
        help="Column / field name(s) to print (space‑separated)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of rows to print (default: all)",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode (good for huge files). Ignores --n if set.",
    )
    return p.parse_args()


def _print_value(value: Any) -> None:
    """Pretty‑print a single value following the original rules."""
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                for key, val in item.items():
                    print(f"{key}: {val}")
            else:
                print(item)
    else:
        print(value)


def main() -> None:
    args = parse_args()

    # Load the dataset. For large files `streaming=True` avoids materializing all rows.
    ds = load_dataset("parquet", data_files=args.input, split="train", streaming=args.stream)

    # Validate requested columns exist
    missing = [f for f in args.field if f not in ds.column_names]
    if missing:
        raise ValueError(
            f"Field(s) {missing} not found. Available columns: {ds.column_names}"
        )

    # Iterate and print.
    rows_seen = 0
    for row in ds:
        for fld in args.field:  # preserve the order provided by the user
            _print_value(row[fld])
        print("=========================================================================")
        rows_seen += 1
        if not args.stream and args.n is not None and rows_seen >= args.n:
            break


if __name__ == "__main__":
    main()