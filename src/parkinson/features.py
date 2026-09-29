"""Feature engineering shared by the experiments.

``PatientFeatures`` turns the raw visit table into a numeric matrix that,
besides the visit's own columns, describes the **patient's other visits**.
The debiased target is a smooth, increasing curve per patient, while each
visit's measured ``on`` / ``off`` is a noisy reading of it, so pooling a
patient's visits is where most of the signal is.

Everything here is computed from input columns only (never ``target``), and
per patient, so the transformer is stateless: it is safe inside grouped CV
and valid on ``X_test`` (each test patient also has 4–12 visits).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

GENE_CODES = {"No Mutation": 0, "LRRK2+": 1, "GBA+": 2, "OTHER+": 3}

BASE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def _patient_trend(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Per-patient least-squares line of ``col`` over ``age``.

    Returns ``(trend, slope)`` aligned on ``df``: the line evaluated at each
    visit's age (also for visits where ``col`` is missing) and its slope.
    Patients with fewer than 2 observed values get their mean and slope 0;
    patients with none get NaN.
    """
    obs = df[col].notna()
    a = df["age"].where(obs)
    v = df[col]
    g = df["patient_id"]
    n = obs.groupby(g).transform("sum")
    mean_a = a.groupby(g).transform("mean")
    mean_v = v.groupby(g).transform("mean")
    da = (a - mean_a).where(obs)
    sxx = (da**2).groupby(g).transform("sum")
    sxy = (da * (v - mean_v)).groupby(g).transform("sum")
    slope = (sxy / sxx).where((n >= 2) & (sxx > 0), 0.0)
    trend = mean_v + slope * (df["age"] - mean_a.fillna(df["age"]))
    return trend.where(n > 0), slope.where(n > 0)


class PatientFeatures(TransformerMixin, BaseEstimator):
    """Visit columns + patient-level aggregates, as a numeric DataFrame.

    Expects the raw visit columns including ``patient_id``; drops ``Index``,
    ``patient_id`` and ``target`` from the output.
    """

    def fit(self, X, y=None):
        # nothing to learn; record the input columns so sklearn / skore see
        # the transformer as fitted
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        df = X.reset_index(drop=True)
        out = df[BASE_COLS].copy()
        g = df["patient_id"]

        out["dur"] = df["age"] - df["age_at_diagnosis"]
        out["cohort_B"] = (df["cohort"] == "B").astype(float)
        out["gene_code"] = df["gene"].map(GENE_CODES)  # NaN = unknown

        # position of the visit in the patient's history
        out["n_visits"] = g.map(g.value_counts())
        out["visit_rank"] = df.groupby("patient_id")["age"].rank()
        out["p_age_mean"] = df.groupby("patient_id")["age"].transform("mean")
        out["age_centered"] = df["age"] - out["p_age_mean"]
        out["p_age_span"] = df.groupby("patient_id")["age"].transform(
            lambda s: s.max() - s.min()
        )

        # patient summaries of the noisy measurements
        for col in ["off", "on", "ledd"]:
            grp = df.groupby("patient_id")[col]
            for stat in ["mean", "median", "min", "max", "count"]:
                out[f"p_{col}_{stat}"] = grp.transform(stat)

        # patient-level linear trend of off / on over age, at this visit's age
        for col in ["off", "on"]:
            trend, slope = _patient_trend(df, col)
            out[f"p_{col}_trend"] = trend
            out[f"p_{col}_slope"] = slope

        out.index = X.index
        return out
