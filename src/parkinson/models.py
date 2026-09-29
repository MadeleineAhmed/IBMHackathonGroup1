"""Model wrappers shared by the experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.isotonic import IsotonicRegression


class PatientSmoother(RegressorMixin, BaseEstimator):
    """Smooth each patient's predictions over age.

    The debiased target is a smooth, increasing curve per patient, but a
    visit-level model predicts every visit separately, so a patient's
    predictions wiggle around that curve. After ``estimator`` predicts,
    this fits, per patient, a polynomial of ``degree`` over ``age`` through
    the predictions (``degree="isotonic"``: best non-decreasing fit) and
    returns ``blend * smoothed + (1 - blend) * raw``.

    Uses only ``patient_id`` and ``age`` from ``X`` at predict time; with
    patient-grouped CV every patient is entirely in train or in validation,
    so nothing leaks across patients.
    """

    def __init__(self, estimator, degree=2, blend=1.0):
        self.estimator = estimator
        self.degree = degree
        self.blend = blend

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        raw = pd.Series(self.estimator_.predict(X), index=X.index)
        smooth = raw.copy()
        for _, idx in X.groupby("patient_id").groups.items():
            age = X.loc[idx, "age"].to_numpy()
            pred = raw.loc[idx].to_numpy()
            if self.degree == "isotonic":
                fit = IsotonicRegression().fit_transform(age, pred)
            elif len(idx) > self.degree and np.unique(age).size > self.degree:
                fit = np.polyval(np.polyfit(age, pred, self.degree), age)
            else:
                continue
            smooth.loc[idx] = fit
        return (self.blend * smooth + (1 - self.blend) * raw).to_numpy()
