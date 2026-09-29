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

import numpy as np
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


# --- 07: pharmacokinetic correction ---------------------------------------

TON_BINS = [-1, 0.5, 1, 1.5, 2, 3, 4, 6, 48]  # hours since intake, ON exam
TOFF_BINS = [-1, 8, 10, 12, 15, 20, 48]  # hours since intake, OFF exam


def _weighted_trend(
    df: pd.DataFrame, values: pd.Series, weights: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-patient weighted least-squares line of ``values`` over ``age``.

    Returns ``(trend, slope, total_weight)`` aligned on ``df``. Patients with
    a single observation (or no age spread) get their weighted mean and
    slope 0; patients with none get NaN.
    """
    obs = values.notna() & weights.notna()
    w = weights.where(obs, 0.0)
    g = df["patient_id"]
    sw = w.groupby(g).transform("sum")
    n = obs.groupby(g).transform("sum")
    a = df["age"]
    mean_a = (w * a).groupby(g).transform("sum") / sw
    mean_v = (w * values.fillna(0)).groupby(g).transform("sum") / sw
    da = a - mean_a
    sxx = (w * da**2).groupby(g).transform("sum")
    sxy = (w * da * (values.fillna(0) - mean_v)).groupby(g).transform("sum")
    slope = (sxy / sxx).where((n >= 2) & (sxx > 0), 0.0)
    trend = mean_v + slope * da
    has = sw > 0
    return trend.where(has), slope.where(has), sw


class PatientPKFeatures(PatientFeatures):
    """``PatientFeatures`` + timing-corrected estimates of the target.

    Learned in ``fit`` from the training fold only:

    - ON ratio curve: median of ``on / target`` per bin of
      ``time_since_intake_on`` (the drug still working lowers the ON score);
    - OFF bias curve: mean of ``off - target`` per bin of
      ``time_since_intake_off``;
    - the residual variance of each corrected reading, used as
      inverse-variance weights.

    ``transform`` turns every reading into a target estimate
    (``on / ratio``, ``off - bias``), combines them per visit, and fits a
    weighted line per patient over age through those estimates.
    """

    def fit(self, X, y):
        super().fit(X, y)
        d = X.assign(_y=np.asarray(y, dtype=float))

        on = d.dropna(subset=["on"])
        on = on[on["_y"] > 0]
        ratio = on["on"] / on["_y"]
        self.on_ratio_ = ratio.groupby(
            pd.cut(on["time_since_intake_on"], TON_BINS), observed=False
        ).median()
        self.on_ratio_na_ = float(ratio[on["time_since_intake_on"].isna()].median())

        off = d.dropna(subset=["off"])
        bias = off["off"] - off["_y"]
        self.off_bias_ = bias.groupby(
            pd.cut(off["time_since_intake_off"], TOFF_BINS), observed=False
        ).mean()
        self.off_bias_na_ = float(bias[off["time_since_intake_off"].isna()].mean())

        e_on, e_off = self._estimates(d)
        self.var_on_ = float(((e_on - d["_y"]) ** 2).mean())
        self.var_off_ = float(((e_off - d["_y"]) ** 2).mean())
        return self

    def _estimates(self, df):
        r = (
            pd.cut(df["time_since_intake_on"], TON_BINS)
            .map(self.on_ratio_)
            .astype(float)
            .fillna(self.on_ratio_na_)
        )
        b = (
            pd.cut(df["time_since_intake_off"], TOFF_BINS)
            .map(self.off_bias_)
            .astype(float)
            .fillna(self.off_bias_na_)
        )
        return df["on"] / r, df["off"] - b

    def transform(self, X):
        out = super().transform(X)
        df = X.reset_index(drop=True)
        e_on, e_off = self._estimates(df)
        w_on = pd.Series(1 / self.var_on_, index=df.index).where(e_on.notna())
        w_off = pd.Series(1 / self.var_off_, index=df.index).where(e_off.notna())
        w = w_on.fillna(0) + w_off.fillna(0)
        est = (e_on.fillna(0) * w_on.fillna(0) + e_off.fillna(0) * w_off.fillna(0)) / w
        est = est.where(w > 0)

        new = pd.DataFrame(index=df.index)
        new["est_on"] = e_on
        new["est_off"] = e_off
        new["est"] = est
        new["p_est_trend"], new["p_est_slope"], new["p_est_weight"] = _weighted_trend(
            df, est, w.where(w > 0)
        )
        new["p_est_mean"] = est.groupby(df["patient_id"]).transform("mean")
        new["p_est_on_trend"], _, _ = _weighted_trend(df, e_on, w_on)
        new["p_est_off_trend"], _, _ = _weighted_trend(df, e_off, w_off)
        new.index = out.index
        return pd.concat([out, new], axis=1)


# --- 09: curved per-patient trend -----------------------------------------


class PatientCurveFeatures(PatientPKFeatures):
    """``PatientPKFeatures`` + a curved per-patient trend and its reliability.

    The true target bends slightly with age (a quadratic fits a patient to
    ~0.2 points, a line to ~1.2). Adds, from the timing-corrected estimates
    ``est``:

    - ``p_est_quad`` / ``p_est_curv``: per-patient quadratic of ``est`` over
      age (patients with at least ``min_obs`` estimates), at the visit's age,
      and its curvature;
    - ``p_est_resid_sd``: spread of ``est`` around the patient's line — how
      noisy this patient's readings are, i.e. how much to trust the trend;
    - ``n_obs_est``: number of visits with at least one reading.
    """

    def __init__(self, min_obs=4):
        self.min_obs = min_obs

    def transform(self, X):
        out = super().transform(X)
        df = X.reset_index(drop=True)
        g = df["patient_id"]
        est = pd.Series(out["est"].to_numpy(), index=df.index)
        obs = est.notna()

        resid = (est - pd.Series(out["p_est_trend"].to_numpy(), index=df.index)).where(obs)
        new = pd.DataFrame(index=df.index)
        new["p_est_resid_sd"] = resid.groupby(g).transform("std")
        new["n_obs_est"] = obs.groupby(g).transform("sum")

        quad = pd.Series(np.nan, index=df.index)
        curv = pd.Series(np.nan, index=df.index)
        for _, idx in df.groupby("patient_id").groups.items():
            m = obs.loc[idx].to_numpy()
            if m.sum() >= self.min_obs:
                age = df.loc[idx, "age"].to_numpy()
                center = age[m].mean()
                coef = np.polyfit(age[m] - center, est.loc[idx].to_numpy()[m], 2)
                quad.loc[idx] = np.polyval(coef, age - center)
                curv.loc[idx] = coef[0]
        new["p_est_quad"] = quad
        new["p_est_curv"] = curv
        new.index = out.index
        return pd.concat([out, new], axis=1)


# --- 12: personal drug response + missing-value patterns -------------------

MISSING_COLS = ["ledd", "on", "off", "time_since_intake_on", "time_since_intake_off"]


class PatientPersonalFeatures(PatientCurveFeatures):
    """``PatientCurveFeatures`` + personal ON calibration + missing patterns.

    **Personal ON calibration.** 39% of the variation of the ON/true ratio
    is between patients (some respond more strongly to levodopa), but the
    07 correction uses one population curve. For each patient, visits where
    both ON and OFF were measured give their personal factor
    ``est_on / est_off`` (median of logs, shrunk toward 1 with
    ``n / (n + shrink)``); ON readings are divided by it before being
    combined. This mostly helps ON-only visits, the hardest ones.

    **Missing-value patterns.** Which of ``ledd``, ``on``, ``off`` and the
    two timings are missing, as a per-visit code and per-patient shares.
    """

    def __init__(self, min_obs=4, shrink=2):
        self.min_obs = min_obs
        self.shrink = shrink

    def transform(self, X):
        out = super().transform(X)
        df = X.reset_index(drop=True)
        g = df["patient_id"]
        e_on, e_off = self._estimates(df)
        new = pd.DataFrame(index=df.index)

        # personal ON factor from visits with both readings
        both = e_on.notna() & e_off.notna() & (e_off > 5) & (e_on > 0)
        log_factor = np.log((e_on / e_off).where(both))
        n_both = both.groupby(g).transform("sum")
        factor = np.exp(
            log_factor.groupby(g).transform("median").fillna(0)
            * n_both / (n_both + self.shrink)
        )
        e_on_p = e_on / factor
        w_on = pd.Series(1 / self.var_on_, index=df.index).where(e_on_p.notna())
        w_off = pd.Series(1 / self.var_off_, index=df.index).where(e_off.notna())
        w = w_on.fillna(0) + w_off.fillna(0)
        est_p = ((e_on_p.fillna(0) * w_on.fillna(0) + e_off.fillna(0) * w_off.fillna(0)) / w).where(w > 0)

        new["p_on_factor"] = factor
        new["n_both"] = n_both
        new["est_on_pers"] = e_on_p
        new["est_pers"] = est_p
        new["p_est_pers_trend"], new["p_est_pers_slope"], _ = _weighted_trend(
            df, est_p, w.where(w > 0)
        )
        quad = pd.Series(np.nan, index=df.index)
        for _, idx in df.groupby("patient_id").groups.items():
            m = est_p.loc[idx].notna().to_numpy()
            if m.sum() >= self.min_obs:
                age = df.loc[idx, "age"].to_numpy()
                center = age[m].mean()
                coef = np.polyfit(age[m] - center, est_p.loc[idx].to_numpy()[m], 2)
                quad.loc[idx] = np.polyval(coef, age - center)
        new["p_est_pers_quad"] = quad

        # missing-value patterns
        miss = df[MISSING_COLS].isna().astype(int)
        new["miss_code"] = (miss * [1, 2, 4, 8, 16]).sum(axis=1)
        for col in MISSING_COLS:
            new[f"p_miss_{col}"] = miss[col].groupby(g).transform("mean")

        new.index = out.index
        return pd.concat([out, new], axis=1)


# --- 14: distance to the patient's real OFF readings ----------------------


class PatientDistanceFeatures(PatientPersonalFeatures):
    """``PatientPersonalFeatures`` + how far each visit is from real OFF data.

    Error analysis of 12: patients with few OFF readings and the last
    visits are the worst. For each visit this adds the age gap to the
    patient's nearest visit with an OFF reading, the corrected OFF estimate
    at that visit, and the age gap to the patient's last visit.
    """

    def transform(self, X):
        out = super().transform(X)
        df = X.reset_index(drop=True)
        _, e_off = self._estimates(df)
        ages = df["age"].to_numpy()
        e_off = e_off.to_numpy()
        dist = np.full(len(df), np.nan)
        nearest = np.full(len(df), np.nan)
        to_last = np.full(len(df), np.nan)
        for _, idx in df.groupby("patient_id").groups.items():
            idx = np.asarray(idx)
            a = ages[idx]
            to_last[idx] = a.max() - a
            has_off = ~np.isnan(e_off[idx])
            if has_off.any():
                gaps = np.abs(a[:, None] - a[has_off][None, :])
                j = gaps.argmin(axis=1)
                dist[idx] = gaps[np.arange(len(a)), j]
                nearest[idx] = e_off[idx][has_off][j]
        new = pd.DataFrame(
            {"dist_nearest_off": dist, "est_off_nearest": nearest, "age_to_last": to_last},
            index=out.index,
        )
        return pd.concat([out, new], axis=1)
