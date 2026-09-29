"""Preliminary smoke tests for the skore library.

Covers the four main entry points used in this project:
  - CrossValidationReport
  - EstimatorReport
  - ComparisonReport
  - TrainTestSplit
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import skore
from skore import (
    ComparisonReport,
    CrossValidationReport,
    EstimatorReport,
    TrainTestSplit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Binary classification dataset (1 000 samples)."""
    X, y = make_classification(
        n_samples=1_000,
        n_features=10,
        n_informative=4,
        random_state=0,
    )
    return X, y


@pytest.fixture
def reg_data():
    """Regression dataset (1 000 samples)."""
    X, y = make_regression(
        n_samples=1_000,
        n_features=10,
        n_informative=4,
        noise=0.1,
        random_state=0,
    )
    return X, y


# ---------------------------------------------------------------------------
# Import / version
# ---------------------------------------------------------------------------

def test_import_and_version():
    """skore is importable and exposes a version string."""
    assert hasattr(skore, "__version__"), "skore.__version__ not found"
    assert isinstance(skore.__version__, str)
    assert skore.__version__  # non-empty


def test_public_api_exports():
    """Key classes are accessible directly from the skore namespace."""
    for name in ("CrossValidationReport", "EstimatorReport", "ComparisonReport", "TrainTestSplit"):
        assert hasattr(skore, name), f"skore.{name} missing from public API"


# ---------------------------------------------------------------------------
# TrainTestSplit
# ---------------------------------------------------------------------------

def test_train_test_split_basic(clf_data):
    """TrainTestSplit.split() returns a generator of (train_idx, test_idx) tuples."""
    X, y = clf_data
    tts = TrainTestSplit(test_size=0.2, random_state=42)
    # split() is an sklearn-style splitter — yields (train_idx, test_idx) index arrays
    splits = list(tts.split(X, y))
    assert len(splits) == 1, "TrainTestSplit should produce exactly one fold"
    train_idx, test_idx = splits[0]

    assert len(train_idx) == 800
    assert len(test_idx) == 200
    # Use indices to slice
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    assert X_train.shape == (800, 10)
    assert X_test.shape == (200, 10)


# ---------------------------------------------------------------------------
# EstimatorReport
# ---------------------------------------------------------------------------

def test_estimator_report_classification(clf_data):
    """EstimatorReport works for a fitted binary classifier."""
    X, y = clf_data
    train_idx, test_idx = next(TrainTestSplit(test_size=0.2, random_state=42).split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=200)
    report = EstimatorReport(clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    summary = report.metrics.summarize().frame()
    assert summary is not None
    assert len(summary) > 0


def test_estimator_report_regression(reg_data):
    """EstimatorReport works for a fitted regressor."""
    X, y = reg_data
    train_idx, test_idx = next(TrainTestSplit(test_size=0.2, random_state=42).split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    reg = LinearRegression()
    report = EstimatorReport(reg, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    summary = report.metrics.summarize().frame()
    assert summary is not None
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# CrossValidationReport
# ---------------------------------------------------------------------------

def test_cross_validation_report(clf_data):
    """CrossValidationReport runs CV and exposes a metrics summary."""
    X, y = clf_data
    clf = LogisticRegression(max_iter=200)
    # `splitter=` accepts an int (number of folds) in v0.26
    report = CrossValidationReport(clf, X, y, splitter=3)

    summary = report.metrics.summarize().frame()
    assert summary is not None
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------

def test_comparison_report(clf_data):
    """ComparisonReport compares two estimator reports."""
    X, y = clf_data
    train_idx, test_idx = next(TrainTestSplit(test_size=0.2, random_state=42).split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    r1 = EstimatorReport(
        LogisticRegression(max_iter=200),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )
    r2 = EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )

    comp = ComparisonReport([r1, r2])
    summary = comp.metrics.summarize().frame()
    assert summary is not None
    # Comparison should have a row/column per estimator
    assert len(summary) > 0
