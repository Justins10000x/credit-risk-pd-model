"""Loading the raw Lending Club file and turning it into a modelling dataset.

The raw file has ~2.2 million loans and ~150 columns. We read it in chunks,
keep only the columns we need and only 36-month loans issued 2007-2015, so
the whole thing fits comfortably in a laptop's memory.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Columns read from the raw file. Grouped by what they are used for.
APPLICATION_COLUMNS = [
    # Known when the customer applies
    "loan_amnt", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "application_type",
]
PRICING_COLUMNS = [
    # Set by Lending Club's own credit model: used only as a benchmark
    "grade", "sub_grade", "int_rate", "installment",
]
OUTCOME_COLUMNS = [
    # Only known after the loan has run: never PD model inputs.
    # Kept for the leakage demonstration (v1) and LGD/EAD (v4).
    "funded_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_fico_range_high", "debt_settlement_flag",
]
KEY_COLUMNS = ["id", "term", "issue_d", "loan_status"]

RAW_COLUMNS = KEY_COLUMNS + APPLICATION_COLUMNS + PRICING_COLUMNS + OUTCOME_COLUMNS

# --- Default definition ----------------------------------------------------
DEFAULT_STATUSES = {"Charged Off", "Default"}
NON_DEFAULT_STATUSES = {"Fully Paid"}
OUTSIDE_POLICY_PREFIX = "Does not meet the credit policy"


def status_group(loan_status: pd.Series) -> pd.Series:
    """Group Lending Club's loan_status values into four buckets."""
    s = loan_status.astype("string").str.strip()
    group = pd.Series("Unresolved", index=s.index, dtype="object")
    group[s.isin(DEFAULT_STATUSES)] = "Defaulted"
    group[s.isin(NON_DEFAULT_STATUSES)] = "Paid off"
    group[s.str.startswith(OUTSIDE_POLICY_PREFIX, na=False)] = "Outside credit policy"
    return group


def define_default(loan_status: pd.Series) -> pd.Series:
    """Target variable: 1 = defaulted, 0 = paid off, NaN = no final outcome.

    'Current', 'Late' and 'In Grace Period' loans have not finished, so we do
    not know whether they will default. They get NaN and are dropped, rather
    than being counted as good loans.
    """
    group = status_group(loan_status)
    target = pd.Series(np.nan, index=loan_status.index)
    target[group == "Defaulted"] = 1.0
    target[group == "Paid off"] = 0.0
    return target


# --- Parsing helpers -------------------------------------------------------
def parse_term(term: pd.Series) -> pd.Series:
    """' 36 months' -> 36."""
    return pd.to_numeric(
        term.astype("string").str.extract(r"(\d+)", expand=False), errors="coerce"
    )


def parse_month(s: pd.Series) -> pd.Series:
    """'Dec-2015' -> Timestamp('2015-12-01')."""
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def parse_percent(s: pd.Series) -> pd.Series:
    """Handle both 13.56 and '13.56%' (older Lending Club files use the latter)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.rstrip("%").str.strip(), errors="coerce")


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, 'n/a' -> NaN."""
    text = s.astype("string").str.strip()
    years = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    years[text.str.startswith("<", na=False)] = 0
    return years.astype(float)


# --- Loading ---------------------------------------------------------------
def _available_columns(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in wanted if c not in header]
    if missing:
        warnings.warn(f"Columns not in raw file, skipped: {missing}")
    return [c for c in wanted if c in header]


def load_status_by_vintage(path: Path = config.RAW_FILE, chunksize: int = 200_000) -> pd.DataFrame:
    """Just issue year, term and status group for every loan (all vintages).

    Used in notebook 01 to show why recent vintages cannot be used yet.
    Read in chunks and stored as small types, so 2.2m loans fit in a few
    hundred MB of memory.
    """
    parts = []
    reader = pd.read_csv(
        path, usecols=["issue_d", "term", "loan_status"], chunksize=chunksize, dtype="string"
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["issue_d"])
        part = pd.DataFrame({
            "issue_year": parse_month(chunk["issue_d"]).dt.year,
            "term_months": parse_term(chunk["term"]),
            "status_group": status_group(chunk["loan_status"]),
        }).dropna(subset=["issue_year"])
        part["issue_year"] = part["issue_year"].astype("int16")
        part["term_months"] = part["term_months"].astype("float32")
        part["status_group"] = part["status_group"].astype("category")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["status_group"] = out["status_group"].astype("category")
    return out


def load_raw_filtered(path: Path = config.RAW_FILE, chunksize: int = 200_000) -> pd.DataFrame:
    """Read the raw file in chunks, keeping 36-month loans issued 2007-2015."""
    usecols = _available_columns(path, RAW_COLUMNS)
    kept = []
    reader = pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, low_memory=False, dtype={"id": "string"}
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["loan_amnt", "issue_d"])  # drops footer/summary rows
        term = parse_term(chunk["term"])
        year = parse_month(chunk["issue_d"]).dt.year
        mask = (
            (term == config.TERM_MONTHS)
            & year.between(config.FIRST_ISSUE_YEAR, config.LAST_ISSUE_YEAR)
        )
        kept.append(chunk.loc[mask])
    return pd.concat(kept, ignore_index=True)


def build_modelling_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean types, define the target and add derived features."""
    df = raw.copy()

    df["issue_date"] = parse_month(df["issue_d"])
    df["issue_year"] = df["issue_date"].dt.year
    df["term_months"] = parse_term(df["term"])
    df["status_group"] = status_group(df["loan_status"])
    df["default"] = define_default(df["loan_status"])

    # Keep only loans with a final outcome
    df = df.loc[df["default"].notna()].copy()
    df["default"] = df["default"].astype(int)

    # Clean individual fields
    df["int_rate"] = parse_percent(df["int_rate"])
    df["revol_util"] = parse_percent(df["revol_util"]).clip(0, 150)
    df["emp_length_years"] = parse_emp_length(df["emp_length"])
    df["dti"] = df["dti"].where(df["dti"] >= 0).clip(upper=100)

    # Derived features (all known at application)
    earliest = parse_month(df["earliest_cr_line"])
    df["credit_history_years"] = (df["issue_date"] - earliest).dt.days / 365.25
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    income = df["annual_inc"].where(df["annual_inc"] > 0)
    df["log_annual_inc"] = np.log(income)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))

    return df.drop(columns=["issue_d", "term", "emp_length"]).reset_index(drop=True)


def load_processed(path: Path = config.PROCESSED_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/make_dataset.py` first."
        )
    return pd.read_parquet(path)


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train / in-time validation / out-of-time test.

    - train + valid: loans issued in config.TRAIN_YEARS, split at random
      (stratified on default so both have the same default rate)
    - oot: loans issued in config.OOT_YEARS, never seen during development
    """
    from sklearn.model_selection import train_test_split

    dev = df.loc[df["issue_year"].isin(config.TRAIN_YEARS)]
    oot = df.loc[df["issue_year"].isin(config.OOT_YEARS)]
    train, valid = train_test_split(
        dev,
        test_size=config.VALID_FRACTION,
        stratify=dev["default"],
        random_state=config.RANDOM_STATE,
    )
    return {"train": train, "valid": valid, "oot": oot}
