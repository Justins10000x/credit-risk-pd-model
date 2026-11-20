import numpy as np
import pandas as pd

from creditrisk.data import define_default, parse_emp_length, parse_percent, parse_term, status_group


def test_default_definition():
    status = pd.Series([
        "Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)",
        "Does not meet the credit policy. Status:Charged Off",
    ])
    target = define_default(status)
    assert target.iloc[0] == 0
    assert target.iloc[1] == 1
    assert target.iloc[2] == 1
    # Unfinished loans and old-policy loans must not be labelled
    assert target.iloc[3:].isna().all()


def test_outside_policy_is_its_own_group():
    s = pd.Series(["Does not meet the credit policy. Status:Fully Paid"])
    assert status_group(s).iloc[0] == "Outside credit policy"


def test_parse_term():
    assert parse_term(pd.Series([" 36 months", " 60 months"])).tolist() == [36, 60]


def test_parse_emp_length():
    out = parse_emp_length(pd.Series(["10+ years", "< 1 year", "3 years", None]))
    assert out.iloc[:3].tolist() == [10, 0, 3]
    assert np.isnan(out.iloc[3])


def test_parse_percent_handles_both_formats():
    assert parse_percent(pd.Series(["13.56%", " 7.5%"])).tolist() == [13.56, 7.5]
    assert parse_percent(pd.Series([13.56, 7.5])).tolist() == [13.56, 7.5]
