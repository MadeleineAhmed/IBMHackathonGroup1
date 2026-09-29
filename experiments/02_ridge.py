"""Step 8: Ridge linear baseline.

Run from the repository root:

    python experiments/02_ridge.py

Use --no-hub to skip the Skore Hub upload.
Use --alpha to tune the regularisation (default 1.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
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
    project.put("02_ridge", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the step 8 Ridge baseline.")
    parser.add_argument(
        "--no-hub",
        action="store_true",
        help="Skip Skore Hub upload.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularisation strength (default 1.0). Try 0.1, 1.0, 10.0.",
    )
    parser.add_argument(
        "--output",
        default="submission_02_ridge.csv",
        help="Submission CSV filename (relative to repo root).",
    )
    args = parser.parse_args()

    root = repo_root()
    X_train, y_train, X_test, sample_submission = read_csvs(root)
    visits = X_train.merge(y_train, on="Index", validate="one_to_one")

    X = visits[FEATURE_COLS]
    y = visits["target"]

    # Dummy (reference) + Ridge cote a cote dans le meme rapport
    dummy = DummyRegressor(strategy="mean")
    ridge = make_pipeline(
        SimpleImputer(strategy="median"),  # Ridge ne gere pas les NaN
        Ridge(alpha=args.alpha),
    )

    print(f"Evaluation dummy vs ridge (alpha={args.alpha})...")
    report = evaluate({"dummy": dummy, "ridge": ridge}, X, y)
    print("RMSE comparison:")
    print(report.metrics.rmse())

    # Submission : Ridge fit sur tout le train, predit sur X_test
    final_model = clone(ridge).fit(X, y)
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
