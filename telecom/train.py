"""Reproducible grouped holdout + grouped CV, with preprocessing inside each fold."""
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telecom.config import ROOT
from telecom.ml import CATEGORICAL, FEATURES, NUMERIC, load_training_data

SEED = 42


def make_pipeline(classifier):
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def metrics(pipeline, x, y):
    predicted = pipeline.predict(x)
    probabilities = pipeline.predict_proba(x)[:, 1]
    return {"precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def train():
    started = time.perf_counter()
    frame = load_training_data()
    x, y = frame[FEATURES], frame.churn
    # Identical predictor rows stay together, even when their labels differ.
    groups = pd.util.hash_pandas_object(x, index=False).astype(str)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(x, y, groups))
    x_train, x_test, y_train, y_test = x.iloc[train_idx], x.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        make_pipeline(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)),
        {"classifier__max_depth": [6, None], "classifier__min_samples_leaf": [1, 4],
         "classifier__class_weight": [None, "balanced"]},
        cv=cv, scoring="f1", refit=True, n_jobs=1, error_score="raise", return_train_score=False
    )
    search.fit(x_train, y_train, groups=groups.iloc[train_idx])
    baseline = make_pipeline(DummyClassifier(strategy="prior", random_state=SEED)).fit(x_train, y_train)
    dataset_sha = hashlib.sha256((ROOT / "data/raw/customer_churn.csv").read_bytes()).hexdigest()
    version = f"uci563-rf-seed{SEED}-{dataset_sha[:8]}"
    report = {
        "dataset": "UCI Iranian Churn (real public data, CC BY 4.0)", "synthetic": False,
        "dataset_sha256": dataset_sha, "seed": SEED, "features": FEATURES,
        "excluded": ["Status", "Age", "Age Group", "Customer Value"],
        "split": "First fold of 5-fold StratifiedGroupKFold; identical feature rows grouped",
        "cv": "3-fold StratifiedGroupKFold on training partition only; refit by F1",
        "train_rows": len(train_idx), "test_rows": len(test_idx), "unique_feature_groups": groups.nunique(),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "threshold": 0.5, "best_params": search.best_params_, "best_cv_f1": float(search.best_score_),
        "baseline": metrics(baseline, x_test, y_test), "random_forest": metrics(search.best_estimator_, x_test, y_test),
        "confusion_matrix_order": "rows actual [0,1], columns predicted [0,1]",
        "model_version": version,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "training_seconds": round(time.perf_counter() - started, 3)
    }
    reports, models = ROOT / "reports", ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    (reports / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "split.json").write_text(json.dumps({"train_indices": train_idx.tolist(), "test_indices": test_idx.tolist()}), encoding="utf-8")
    pd.DataFrame(search.cv_results_).to_csv(reports / "grid_search.csv", index=False)
    joblib.dump({"pipeline": search.best_estimator_, "model_version": version, "features": FEATURES,
                 "dataset_sha256": dataset_sha}, models / "churn_pipeline.joblib")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
