"""Model definitions.

v1 baseline: logistic regression inside a scikit-learn Pipeline, so that
every preprocessing step (imputation, scaling, encoding) is learned on the
training data only and then applied unchanged to validation and test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import assert_no_leakage


def build_logistic_pipeline(numeric: list[str], categorical: list[str], C: float = 1.0) -> Pipeline:
    assert_no_leakage(numeric + categorical)

    numeric_steps = Pipeline([
        # Median fill, plus a 0/1 column saying "this was missing":
        # missingness itself can carry risk information.
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        # Put features on the same scale so coefficients are comparable
        # and the regularisation treats them equally.
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Rare categories (<1% of loans) are grouped together; categories
        # never seen in training are mapped to that group too.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])
    # No class_weight="balanced": we want predicted probabilities that match
    # real default rates (calibration), not just a good ranking.
    model = LogisticRegression(C=C, max_iter=2000)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients with readable feature names, sorted by size."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    # exp(coefficient) = how the odds of default multiply when the feature
    # rises by one standard deviation (or when a category is present).
    table["odds_ratio"] = np.exp(coefs)
    table["abs"] = table["coefficient"].abs()
    return table.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
