"""How good is a PD model? Two separate questions.

1. Discrimination (ranking): does the model give defaulters higher PDs than
   non-defaulters? Measured by AUC, Gini and KS.
2. Calibration (level): when the model says 10%, do about 10% default?
   Measured by comparing average PD with the observed default rate, the
   Brier score, and a calibration table by decile.

Banks need both: ranking decides who gets a loan, the level feeds expected
loss, provisions and capital.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def gini(y_true, score) -> float:
    """Gini = 2 * AUC - 1. Banks quote Gini; data scientists quote AUC."""
    return 2 * roc_auc_score(y_true, score) - 1


def ks_statistic(y_true, score) -> float:
    """Largest gap between the cumulative score distributions of defaulters
    and non-defaulters (equivalently, max of TPR - FPR on the ROC curve)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def performance_summary(y_true, pd_hat) -> dict:
    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)
    auc = roc_auc_score(y_true, pd_hat)
    return {
        "loans": len(y_true),
        "observed_default_rate": y_true.mean(),
        "mean_predicted_pd": pd_hat.mean(),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, pd_hat),
        "brier": brier_score_loss(y_true, pd_hat),
    }


def summary_table(results: dict[str, tuple]) -> pd.DataFrame:
    """results = {"train": (y, pd_hat), "valid": (...), "oot": (...)}"""
    return pd.DataFrame({name: performance_summary(y, p) for name, (y, p) in results.items()}).T


def calibration_table(y_true, pd_hat, n_bins: int = 10) -> pd.DataFrame:
    """Sort loans into deciles of predicted PD; compare predicted vs actual."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    df["bin"] = pd.qcut(df["pd"].rank(method="first"), n_bins, labels=False) + 1
    table = df.groupby("bin").agg(
        loans=("y", "size"),
        mean_predicted_pd=("pd", "mean"),
        observed_default_rate=("y", "mean"),
    )
    table["ratio_obs_to_pred"] = table["observed_default_rate"] / table["mean_predicted_pd"]
    return table


def default_rate_by(df: pd.DataFrame, column: str, target: str = "default", bins=None) -> pd.DataFrame:
    """Univariate view: loans and default rate per category or per bin."""
    key = pd.cut(df[column], bins) if bins is not None else df[column]
    return (
        df.groupby(key, observed=True)[target]
        .agg(loans="size", default_rate="mean")
        .assign(share_of_loans=lambda t: t["loans"] / t["loans"].sum())
    )


def sub_grade_to_rank(sub_grade: pd.Series) -> pd.Series:
    """'A1' -> 1 ... 'G5' -> 35, so Lending Club's grade can be scored like a PD."""
    s = sub_grade.astype("string")
    letter = s.str[0].map({c: i for i, c in enumerate("ABCDEFG")})
    number = pd.to_numeric(s.str[1], errors="coerce")
    return (letter * 5 + number).astype(float)


# --- Plots -----------------------------------------------------------------
def plot_roc(results: dict[str, tuple], ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    for name, (y, p) in results.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="random")
    ax.set_xlabel("False positive rate (good loans flagged)")
    ax.set_ylabel("True positive rate (defaults caught)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax


def plot_calibration(table: pd.DataFrame, title: str = "Calibration", ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    hi = max(table["mean_predicted_pd"].max(), table["observed_default_rate"].max()) * 1.05
    ax.plot([0, hi], [0, hi], linestyle="--", color="grey", linewidth=1, label="perfect")
    ax.plot(table["mean_predicted_pd"], table["observed_default_rate"], marker="o", label="model")
    ax.set_xlabel("Mean predicted PD (per decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax
