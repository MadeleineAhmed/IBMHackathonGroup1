"""Data loading, patient-grouped CV splits and submission writing.

Shared by every experiment in ``experiments/`` so they all read the
same tables, split patients the same way and write submissions in the
format Kaggle expects (``Index,target``, one row per test visit).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinson import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

ID_COLS = ["Index", "patient_id"]
TARGET = "target"

NUMERIC_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
CATEGORICAL_COLS = ["cohort", "gene"]

N_SPLITS = 5


def load_visits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(visits, X_test)``.

    ``visits`` is ``X_train`` merged with ``y_train`` on ``Index`` (one row
    per training visit, with the ``target`` column). ``X_test`` is returned
    unchanged.
    """
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    visits = X_train.merge(y_train, on="Index", validate="one_to_one")
    return visits, X_test


def grouped_cv_splits(visits: pd.DataFrame, n_splits: int = N_SPLITS) -> list:
    """Precomputed ``(train_idx, test_idx)`` pairs, grouped by ``patient_id``.

    skore calls ``splitter.split(X, y)`` without ``groups=``, so the pairs
    are computed here and passed as ``evaluate(..., splitter=cv_splits)``.
    A patient's visits are never split between train and validation,
    mirroring the Kaggle test set (patients disjoint from train).
    """
    return list(
        GroupKFold(n_splits=n_splits).split(
            visits, visits[TARGET], groups=visits["patient_id"]
        )
    )


def grouped_holdout(visits: pd.DataFrame) -> tuple:
    """One patient-grouped ``(train_idx, test_idx)`` split (~80/20 patients).

    Used to build the ``EstimatorReport`` whose Hub URL goes in the Kaggle
    Submission Description (the rules ask for an EstimatorReport URL); it
    is the first fold of :func:`grouped_cv_splits`.
    """
    return grouped_cv_splits(visits)[0]


def write_submission(X_test: pd.DataFrame, predictions, name: str) -> Path:
    """Write ``submission_<name>.csv`` at the repo root and return its path.

    ``Index`` comes from ``X_test`` itself so each prediction stays on its
    visit; the file is checked against ``sample_submission.csv``.
    """
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = X_test[["Index"]].copy()
    submission[TARGET] = predictions
    if set(submission["Index"]) != set(sample["Index"]) or len(submission) != len(sample):
        raise ValueError("Submission Index does not match sample_submission.csv")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains NaN predictions")
    path = PROJECT_ROOT / f"submission_{name}.csv"
    submission.to_csv(path, index=False)
    return path
