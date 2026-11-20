import numpy as np
import pandas as pd
import pytest

from creditrisk.evaluation import calibration_table, gini, ks_statistic, sub_grade_to_rank
from creditrisk.features import MODEL_FEATURES, assert_no_leakage


def test_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert gini(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_calibration_table_bins_cover_every_loan():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 1000)
    y = rng.binomial(1, p)
    table = calibration_table(y, p, n_bins=10)
    assert table["loans"].sum() == 1000
    assert table["mean_predicted_pd"].is_monotonic_increasing


def test_sub_grade_rank():
    assert sub_grade_to_rank(pd.Series(["A1", "A5", "B1", "G5"])).tolist() == [1, 5, 6, 35]


def test_model_features_are_clean():
    assert_no_leakage(MODEL_FEATURES)


def test_leakage_is_blocked():
    with pytest.raises(ValueError):
        assert_no_leakage(["fico", "recoveries"])
    with pytest.raises(ValueError):
        assert_no_leakage(["fico", "sub_grade"])
