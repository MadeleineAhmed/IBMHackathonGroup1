"""
scripts/env_check.py — Sanity check for the skore hackathon environment.

Run with:  ./run.sh scripts/env_check.py
       or: ./run.sh -m pytest tests -v   (for full test suite)
"""

import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  ✓  {label:<30} {result}")
    except Exception as e:
        print(f"  ✗  {label:<30} FAILED — {e}")
        return False
    return True


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# Python runtime
# ---------------------------------------------------------------------------
section("Python runtime")
check("executable", lambda: sys.executable)
check("version", lambda: sys.version.split()[0])


# ---------------------------------------------------------------------------
# Core data science stack
# ---------------------------------------------------------------------------
section("Core stack")

import numpy as np
check("numpy", lambda: np.__version__)

import pandas as pd
check("pandas", lambda: pd.__version__)

import scipy
check("scipy", lambda: scipy.__version__)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on any system
check("matplotlib", lambda: matplotlib.__version__)

import sklearn
check("scikit-learn", lambda: sklearn.__version__)


# ---------------------------------------------------------------------------
# skore + companions
# ---------------------------------------------------------------------------
section("skore")

import skore
check("skore version", lambda: skore.__version__)

from skore import CrossValidationReport, EstimatorReport, ComparisonReport, TrainTestSplit
check("CrossValidationReport", lambda: "importable")
check("EstimatorReport", lambda: "importable")
check("ComparisonReport", lambda: "importable")
check("TrainTestSplit", lambda: "importable")

import skrub
check("skrub", lambda: skrub.__version__)


# ---------------------------------------------------------------------------
# Quick end-to-end smoke test
# ---------------------------------------------------------------------------
section("End-to-end smoke test")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=500, n_features=8, n_informative=4, random_state=0)
train_idx, test_idx = next(TrainTestSplit(test_size=0.2, random_state=0).split(X, y))

report = EstimatorReport(
    LogisticRegression(max_iter=200),
    X_train=X[train_idx], y_train=y[train_idx],
    X_test=X[test_idx],   y_test=y[test_idx],
)
summary = report.metrics.summarize().frame()
check("EstimatorReport smoke test", lambda: f"{len(summary)} metric(s) returned")

cv_report = CrossValidationReport(LogisticRegression(max_iter=200), X, y, splitter=3)
cv_summary = cv_report.metrics.summarize().frame()
check("CrossValidationReport smoke test", lambda: f"{len(cv_summary)} metric(s) returned")

print("\n  All checks complete.\n")
