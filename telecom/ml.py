from contextlib import closing
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from pydantic import ValidationError

from telecom.config import ROOT
from telecom.errors import AppError, CustomerNotFound
from telecom.schemas import Customer, CustomerInput, Prediction

NUMERIC = ["call_failures", "subscription_months", "charge_band", "seconds_of_use", "calls", "sms", "distinct_called_numbers"]
CATEGORICAL = ["complaints", "tariff_plan"]
FEATURES = NUMERIC + CATEGORICAL
RENAME = {
    "Call  Failure": "call_failures", "Call Failure": "call_failures",
    "Complains": "complaints", "Subscription  Length": "subscription_months",
    "Subscription Length": "subscription_months", "Charge  Amount": "charge_band",
    "Charge Amount": "charge_band", "Seconds of Use": "seconds_of_use",
    "Frequency of use": "calls", "Frequency of SMS": "sms",
    "Distinct Called Numbers": "distinct_called_numbers", "Tariff Plan": "tariff_plan", "Churn": "churn"
}


def load_training_data(path: Path = ROOT / "data/raw/customer_churn.csv") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    frame = raw.rename(columns=RENAME)[FEATURES + ["churn"]].copy()
    for column in NUMERIC + ["churn"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["complaints"] = frame["complaints"].map({0: "no", 1: "yes"})
    frame["tariff_plan"] = frame["tariff_plan"].map({1: "pay_as_you_go", 2: "contract"})
    if frame[CATEGORICAL].isna().any().any() or not set(frame.churn.unique()) <= {0, 1}:
        raise ValueError("Unexpected dataset categories or labels.")
    return frame


class CustomerStore:
    def __init__(self, path: Path = ROOT / "data/customers.sqlite3"):
        self.path = path.resolve()

    def get(self, customer_id: str) -> Customer:
        customer_id = CustomerInput(customer_id=customer_id).customer_id
        if not self.path.exists():
            raise AppError("Initialise the fictional customer database: python -m telecom.init_db", "database_unavailable")
        try:
            # Separate connection per operation: safe for FastAPI's worker threads.
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    """SELECT c.customer_id, c.display_name, c.fictional, c.subscription_months,
                              c.charge_band, c.tariff_plan, u.call_failures, u.complaints,
                              u.seconds_of_use, u.calls, u.sms, u.distinct_called_numbers
                       FROM customers AS c
                       INNER JOIN usage_metrics AS u ON u.customer_id = c.customer_id
                       WHERE c.customer_id = ?""", (customer_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise AppError("The local customer database could not be read. Run python -m telecom.init_db.", "database_unavailable") from exc
        if row is None:
            raise CustomerNotFound(customer_id)
        try:
            return Customer.model_validate(dict(row))
        except ValidationError:
            raise AppError("The local database contains an invalid customer record.", "database_invalid") from None


class Predictor:
    def __init__(self, path: Path = ROOT / "models/churn_pipeline.joblib"):
        if not path.exists():
            raise AppError("Train the churn model first: python -m telecom.train", "model_unavailable")
        # Only load model files created locally by this project; joblib is not safe for untrusted files.
        try:
            self.bundle = joblib.load(path)
            if (not isinstance(self.bundle, dict) or self.bundle.get("features") != FEATURES
                    or not hasattr(self.bundle.get("pipeline"), "predict_proba")
                    or not isinstance(self.bundle.get("model_version"), str)):
                raise ValueError("Invalid local model bundle")
        except Exception:
            raise AppError("Saved churn model is invalid. Rebuild it: python -m telecom.train", "model_unavailable") from None

    def predict(self, customer: Customer) -> Prediction:
        row = pd.DataFrame([customer.model_dump()])[FEATURES]
        probability = float(self.bundle["pipeline"].predict_proba(row)[0, 1])
        return Prediction(customer_id=customer.customer_id, churn_probability=probability,
                          predicted_churn=probability >= 0.5, model_version=self.bundle["model_version"])


