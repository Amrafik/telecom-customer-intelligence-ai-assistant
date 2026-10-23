import json
import sqlite3
from contextlib import closing
from pathlib import Path

from telecom.config import ROOT
from telecom.schemas import Customer

DDL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    fictional INTEGER NOT NULL CHECK (fictional = 1),
    subscription_months INTEGER NOT NULL CHECK (subscription_months BETWEEN 0 AND 600),
    charge_band INTEGER NOT NULL CHECK (charge_band BETWEEN 0 AND 9),
    tariff_plan TEXT NOT NULL CHECK (tariff_plan IN ('pay_as_you_go', 'contract'))
);
CREATE TABLE IF NOT EXISTS usage_metrics (
    customer_id TEXT PRIMARY KEY REFERENCES customers(customer_id),
    call_failures INTEGER NOT NULL CHECK (call_failures >= 0),
    complaints TEXT NOT NULL CHECK (complaints IN ('yes', 'no')),
    seconds_of_use INTEGER NOT NULL CHECK (seconds_of_use >= 0),
    calls INTEGER NOT NULL CHECK (calls >= 0),
    sms INTEGER NOT NULL CHECK (sms >= 0),
    distinct_called_numbers INTEGER NOT NULL CHECK (distinct_called_numbers >= 0)
);
"""


def initialise_database(path: Path = ROOT / "data/customers.sqlite3",
                        seed_path: Path = ROOT / "data/customers.json"):
    customers = [Customer.model_validate(row) for row in json.loads(seed_path.read_text(encoding="utf-8"))]
    if len({row.customer_id for row in customers}) != len(customers):
        raise ValueError("Duplicate customer ID in seed records.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(DDL)
        for row in customers:
            connection.execute(
                """INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(customer_id) DO NOTHING""",
                (row.customer_id, row.display_name, int(row.fictional), row.subscription_months,
                 row.charge_band, row.tariff_plan)
            )
            connection.execute(
                """INSERT INTO usage_metrics VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(customer_id) DO NOTHING""",
                (row.customer_id, row.call_failures, row.complaints, row.seconds_of_use,
                 row.calls, row.sms, row.distinct_called_numbers)
            )
    print(f"SQLite ready: {path}; {len(customers)} fictional seed records (existing rows preserved).")


if __name__ == "__main__":
    initialise_database()
