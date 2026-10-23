import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from telecom.config import ROOT
from telecom.ml import FEATURES, CustomerStore, Predictor, load_training_data
from telecom.train import make_pipeline, metrics


@pytest.mark.integration
def test_split_and_cv_keep_identical_features_together():
    frame = load_training_data()
    split = json.loads((ROOT / "reports/split.json").read_text())
    train_idx, test_idx = split["train_indices"], split["test_indices"]
    assert set(train_idx).isdisjoint(test_idx)
    assert set(train_idx) | set(test_idx) == set(range(len(frame)))
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False)
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    training = frame.iloc[train_idx]
    train_groups = groups.iloc[train_idx]
    for fit, validation in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(training, training.churn, train_groups):
        assert set(train_groups.iloc[fit]).isdisjoint(set(train_groups.iloc[validation]))


@pytest.mark.integration
def test_saved_benchmark_matches_real_holdout_predictions():
    frame = load_training_data()
    report = json.loads((ROOT / "reports/benchmark.json").read_text())
    split = json.loads((ROOT / "reports/split.json").read_text())
    bundle = joblib.load(ROOT / "models/churn_pipeline.joblib")
    test = frame.iloc[split["test_indices"]]
    assert metrics(bundle["pipeline"], test[FEATURES], test.churn) == report["random_forest"]
    assert np.asarray(report["random_forest"]["confusion_matrix"]).sum() == len(test)
    assert "churn" not in bundle["features"] and "customer_id" not in bundle["features"]
    assert report["random_forest"]["f1"] > report["baseline"]["f1"]


@pytest.mark.integration
def test_seeded_refit_reproduces_saved_probabilities():
    frame = load_training_data()
    split = json.loads((ROOT / "reports/split.json").read_text())
    pipeline = joblib.load(ROOT / "models/churn_pipeline.joblib")["pipeline"]
    train, test = frame.iloc[split["train_indices"]], frame.iloc[split["test_indices"]]
    refit = clone(pipeline).fit(train[FEATURES], train.churn)
    np.testing.assert_allclose(pipeline.predict_proba(test[FEATURES]), refit.predict_proba(test[FEATURES]), atol=1e-12)


def test_preprocessing_imputes_from_training_and_handles_unknown_category():
    row = CustomerStore().get("CUST-001").model_dump()
    frame = pd.DataFrame([row, row, row, row])[FEATURES]
    frame["seconds_of_use"] = [10.0, 20.0, 30.0, np.nan]
    pipeline = make_pipeline(RandomForestClassifier(n_estimators=3, random_state=42)).fit(frame, [0, 1, 0, 1])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    from telecom.ml import NUMERIC
    assert imputer.statistics_[NUMERIC.index("seconds_of_use")] == 20.0
    future = frame.iloc[[0]].copy()
    future["tariff_plan"] = "unseen_plan"
    future["seconds_of_use"] = np.nan
    assert np.isfinite(pipeline.predict_proba(future)).all()


def test_prediction_is_valid_and_stable(assistant):
    row = assistant.customers.get("CUST-002")
    first, second = assistant.predictor().predict(row), assistant.predictor().predict(row)
    assert 0 <= first.churn_probability <= 1 and first == second
    assert first.predicted_churn == (first.churn_probability >= first.threshold)


def test_missing_model_has_actionable_error(tmp_path):
    from telecom.errors import AppError
    with pytest.raises(AppError, match="Train the churn model"):
        Predictor(tmp_path / "missing.joblib")


@pytest.mark.parametrize("contents", [None, {"features": [], "pipeline": "invalid", "model_version": "x"}])
def test_corrupt_or_incompatible_model_is_actionable(tmp_path, contents):
    from telecom.errors import AppError
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"invalid model") if contents is None else joblib.dump(contents, path)
    with pytest.raises(AppError) as caught:
        Predictor(path)
    assert caught.value.code == "model_unavailable"


def test_baseline_and_artifact_hashes_are_reproducible():
    import hashlib
    from sklearn.dummy import DummyClassifier
    report = json.loads((ROOT / "reports/benchmark.json").read_text())
    split = json.loads((ROOT / "reports/split.json").read_text())
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    frame = load_training_data()
    train, test = frame.iloc[split["train_indices"]], frame.iloc[split["test_indices"]]
    pipeline = make_pipeline(DummyClassifier(strategy="prior", random_state=42)).fit(train[FEATURES], train.churn)
    assert metrics(pipeline, test[FEATURES], test.churn) == report["baseline"]
    digest = hashlib.sha256((ROOT / "data/raw/customer_churn.csv").read_bytes()).hexdigest()
    bundle = joblib.load(ROOT / "models/churn_pipeline.joblib")
    assert digest == provenance["sha256"] == report["dataset_sha256"] == bundle["dataset_sha256"]


@pytest.mark.integration
def test_full_grid_search_reproduces_original_benchmark_in_temporary_output(tmp_path, monkeypatch):
    import shutil
    from telecom.train import train
    expected = json.loads((ROOT / "reports/benchmark.json").read_text())
    target = tmp_path / "data/raw/customer_churn.csv"
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data/raw/customer_churn.csv", target)
    monkeypatch.setattr("telecom.train.ROOT", tmp_path)
    actual = train()  # Uses the same immutable input; writes only temporary models/reports.
    for field in ["best_params", "best_cv_f1", "baseline", "random_forest", "train_rows", "test_rows"]:
        assert actual[field] == expected[field]
    grid = pd.read_csv(tmp_path / "reports/grid_search.csv")
    original_grid = pd.read_csv(ROOT / "reports/grid_search.csv")
    assert len(grid) == len(original_grid) == 8
    np.testing.assert_allclose(grid["mean_test_score"], original_grid["mean_test_score"], atol=1e-12)
