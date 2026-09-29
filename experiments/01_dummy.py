"""Step 7: dummy mean baseline.

Run from the repository root:

    python experiments/01_dummy.py

Use --no-hub if you only want the local check and submission file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from skore import Project, evaluate, login

from parkinson.hub import load_skore_credentials


FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csvs(root: Path):
    data_dir = root / "data"
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return X_train, y_train, X_test, sample_submission


def push_to_hub(report) -> None:
    cfg = load_skore_credentials()
    login(mode="hub")
    project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
    project.put("01_dummy", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the step 7 dummy baseline.")
    parser.add_argument(
        "--no-hub",
        action="store_true",
        help="Skip Skore Hub upload and only run the local baseline.",
    )
    parser.add_argument(
        "--output",
        default="submission_01_dummy.csv",
        help="Submission CSV path, relative to the repository root by default.",
    )
    args = parser.parse_args()

    root = repo_root()
    X_train, y_train, X_test, sample_submission = read_csvs(root)
    visits = X_train.merge(y_train, on="Index", validate="one_to_one")

    X = visits[FEATURE_COLS]
    y = visits["target"]

    dummy = DummyRegressor(strategy="mean")
    report = evaluate(dummy, X, y)

    print(f"Training target mean: {y.mean():.6f}")
    print("Dummy RMSE:")
    print(report.metrics.rmse())

    final_model = clone(dummy).fit(X, y)
    submission = sample_submission[["Index"]].copy()
    submission["target"] = final_model.predict(X_test[FEATURE_COLS])

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    submission.to_csv(output_path, index=False)
    print(f"Wrote {output_path.relative_to(root)}")

    if args.no_hub:
        print("Skipped Skore Hub upload (--no-hub).")
    else:
        push_to_hub(report)


if __name__ == "__main__":
    main()