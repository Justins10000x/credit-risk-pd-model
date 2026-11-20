"""Build the modelling dataset from the raw Lending Club file.

Usage (from the repo root):
    python scripts/make_dataset.py
    python scripts/make_dataset.py --raw path/to/accepted_2007_to_2018Q4.csv.gz

Reads ~2.2m loans in chunks, keeps 36-month loans issued 2007-2015 that have
a final outcome, and writes a much smaller Parquet file to data/processed/.
"""

import argparse
import time
from pathlib import Path

from creditrisk import config
from creditrisk.data import build_modelling_dataset, load_raw_filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RAW_FILE)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_FILE)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(
            f"Raw file not found: {args.raw}\n"
            "Download accepted_2007_to_2018Q4.csv.gz from Kaggle into data/raw/."
        )

    start = time.time()
    print(f"Reading {args.raw} in chunks (a few minutes on a laptop)...")
    raw = load_raw_filtered(args.raw)
    print(f"  36-month loans issued {config.FIRST_ISSUE_YEAR}-{config.LAST_ISSUE_YEAR}: {len(raw):,}")

    df = build_modelling_dataset(raw)
    dropped = len(raw) - len(df)
    print(f"  Dropped {dropped:,} loans with no final outcome or outside credit policy")
    print(f"  Modelling dataset: {len(df):,} loans, default rate {df['default'].mean():.2%}")
    print("\n  Loans and default rate by issue year:")
    print(df.groupby("issue_year")["default"].agg(loans="size", default_rate="mean")
            .to_string(formatters={"default_rate": "{:.2%}".format}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nSaved {args.out} ({size_mb:.0f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
