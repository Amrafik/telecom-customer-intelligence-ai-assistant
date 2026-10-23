import json
import sqlite3

import pytest
from pydantic import ValidationError

from telecom.config import ROOT
from telecom.errors import AppError, CustomerNotFound
from telecom.init_db import initialise_database
from telecom.ml import CustomerStore


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "customers.sqlite3"
    initialise_database(path)
    return path


def test_join_reads_profile_and_usage_from_sql(database):
    # Mutate both tables: a JSON-only or single-table lookup cannot pass this test.
    with sqlite3.connect(database) as db:
        db.execute("UPDATE customers SET subscription_months = ? WHERE customer_id = ?", (50, "CUST-001"))
        db.execute("UPDATE usage_metrics SET calls = ? WHERE customer_id = ?", (777, "CUST-001"))
    row = CustomerStore(database).get("CUST-001")
    assert row.subscription_months == 50 and row.calls == 777
    assert row.display_name == "Demo Customer A" and row.fictional


def test_seed_is_idempotent_and_preserves_existing_rows(database):
    with sqlite3.connect(database) as db:
        db.execute("UPDATE customers SET display_name = ? WHERE customer_id = ?", ("Edited demo name", "CUST-001"))
    initialise_database(database)
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM customers").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM usage_metrics").fetchone()[0] == 6
    assert CustomerStore(database).get("CUST-001").display_name == "Edited demo name"


@pytest.mark.parametrize("bad_id", ["CUST-001' OR 1=1 --", "CUST-001; DROP TABLE customers;", "CUST-1", ""])
def test_invalid_ids_rejected_and_tables_survive(database, bad_id):
    with pytest.raises(ValidationError):
        CustomerStore(database).get(bad_id)
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM customers").fetchone()[0] == 6


def test_sql_is_parameterized(database, monkeypatch):
    real_connect = sqlite3.connect
    calls = []

    class ConnectionProxy:
        def __init__(self, connection):
            object.__setattr__(self, "connection", connection)
        def __setattr__(self, name, value):
            setattr(self.connection, name, value)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.connection.close()
        def close(self):
            self.connection.close()
        def execute(self, sql, params):
            calls.append((sql, params))
            return self.connection.execute(sql, params)

    monkeypatch.setattr("telecom.ml.sqlite3.connect", lambda *a, **kw: ConnectionProxy(real_connect(*a, **kw)))
    CustomerStore(database).get("CUST-001")
    sql, params = calls[0]
    assert "JOIN usage_metrics" in sql and "WHERE c.customer_id = ?" in sql
    assert "CUST-001" not in sql and params == ("CUST-001",)


def test_foreign_key_and_basic_constraints(database):
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO usage_metrics VALUES (?, ?, ?, ?, ?, ?, ?)", ("CUST-999", 1, "no", 10, 1, 1, 1))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE customers SET charge_band = -1 WHERE customer_id = ?", ("CUST-001",))


def test_missing_customer_and_database(database, tmp_path):
    with pytest.raises(CustomerNotFound):
        CustomerStore(database).get("CUST-999")
    with pytest.raises(AppError, match="Initialise"):
        CustomerStore(tmp_path / "missing.sqlite3").get("CUST-001")


def test_seed_validation_rejects_nonfictional_customer(tmp_path):
    rows = json.loads((ROOT / "data/customers.json").read_text())
    rows[0]["fictional"] = False
    seed = tmp_path / "bad.json"
    seed.write_text(json.dumps(rows))
    with pytest.raises(ValidationError):
        initialise_database(tmp_path / "bad.sqlite3", seed)


@pytest.mark.parametrize("customer_id", ["CUST-001", "CUST-999"])
def test_lookup_closes_connection_on_success_or_missing_record(database, monkeypatch, customer_id):
    real_connect = sqlite3.connect
    opened = []
    def tracked_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection
    monkeypatch.setattr("telecom.ml.sqlite3.connect", tracked_connect)
    try:
        CustomerStore(database).get(customer_id)
    except CustomerNotFound:
        assert customer_id == "CUST-999"
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_invalid_database_record_is_not_mislabeled_as_bad_tool_input(database):
    with sqlite3.connect(database) as db:
        db.execute("UPDATE usage_metrics SET call_failures = 1001 WHERE customer_id = ?", ("CUST-001",))
    with pytest.raises(AppError) as caught:
        CustomerStore(database).get("CUST-001")
    assert caught.value.code == "database_invalid"


def test_all_saved_sql_records_match_fictional_seed():
    from contextlib import closing
    from telecom.schemas import Customer
    rows = [Customer.model_validate(row) for row in json.loads((ROOT / "data/customers.json").read_text())]
    with closing(sqlite3.connect(ROOT / "data/customers.sqlite3")) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM customers").fetchone()[0] == len(rows)
        assert connection.execute("SELECT count(*) FROM usage_metrics").fetchone()[0] == len(rows)
    for row in rows:
        assert CustomerStore().get(row.customer_id) == row
