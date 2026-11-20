"""Project-wide paths and modelling choices.

Keeping these in one place means every notebook and script uses the same
sample definition, which is what makes results reproducible.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = DATA_RAW / "accepted_2007_to_2018Q4.csv.gz"
PROCESSED_FILE = DATA_PROCESSED / "loans_36m_2007_2015.parquet"

# --- Sample definition -----------------------------------------------------
# Only 36-month loans: the data ends in late 2018, so a 36-month loan issued
# by December 2015 has had its full term to either repay or default.
TERM_MONTHS = 36
FIRST_ISSUE_YEAR = 2007
LAST_ISSUE_YEAR = 2015

# --- Validation design -----------------------------------------------------
# Train on earlier vintages, keep the latest vintage as an out-of-time (OOT)
# test set. A random slice of the training years is held back as an
# in-time validation set.
TRAIN_YEARS = range(2007, 2015)  # 2007-2014 inclusive
OOT_YEARS = range(2015, 2016)    # 2015
VALID_FRACTION = 0.2
RANDOM_STATE = 42
