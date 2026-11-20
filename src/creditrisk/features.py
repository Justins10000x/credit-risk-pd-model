"""Which columns the PD model may use, and which it must never use.

The rule: a PD model scores a customer at the moment they apply. Any column
whose value is only known after that moment is leakage, however predictive
it looks.
"""

from __future__ import annotations

# --- Allowed: known at application ----------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt",
    "log_annual_inc",
    "loan_to_income",
    "dti",
    "fico",
    "emp_length_years",
    "credit_history_years",
    "inq_last_6mths",
    "delinq_2yrs",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_util",
    "log_revol_bal",
    "mort_acc",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Excluded: Lending Club's own pricing ---------------------------------
# Known at application, so not leakage in the timing sense. But they are the
# output of Lending Club's credit model. Feeding them in would mean our model
# mostly copies theirs. We use sub_grade as a benchmark instead.
PRICING_COLUMNS = {
    "grade": "Lending Club's risk grade (A-G), output of their model",
    "sub_grade": "Finer version of grade (A1-G5)",
    "int_rate": "Priced from sub_grade, so carries the same information",
    "installment": "Calculated from loan amount, term and int_rate",
}

# --- Excluded: fairness ---------------------------------------------------
# Could act as a proxy for protected characteristics; also high cardinality.
FAIRNESS_EXCLUDED = {
    "addr_state": "Geography can proxy for protected characteristics",
}

# --- Forbidden: leakage (only known after the loan runs) ------------------
LEAKAGE_COLUMNS = {
    "total_pymnt": "Total paid back so far: low if the borrower defaulted",
    "total_rec_prncp": "Principal repaid: a defaulter never repays it all",
    "total_rec_int": "Interest received to date",
    "recoveries": "Money collected AFTER a charge-off: only non-zero for defaults",
    "collection_recovery_fee": "Fee for collecting recoveries: only for defaults",
    "last_pymnt_d": "Date of last payment: early for defaulters",
    "last_fico_range_high": "Credit score pulled later in the loan's life",
    "debt_settlement_flag": "Borrower settled a defaulted debt",
    "out_prncp": "Principal still outstanding today",
    "loan_status": "This IS the outcome",
}


def assert_no_leakage(features: list[str]) -> None:
    """Fail loudly if a forbidden column sneaks into a feature list."""
    forbidden = set(LEAKAGE_COLUMNS) | set(PRICING_COLUMNS) | set(FAIRNESS_EXCLUDED)
    bad = sorted(set(features) & forbidden)
    if bad:
        raise ValueError(f"Forbidden columns in feature list: {bad}")
