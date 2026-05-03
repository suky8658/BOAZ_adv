import json

from sql_agent.sql_agent import (
    get_metadata_context,
    is_safe_datamart_table_name,
    is_safe_select_sql,
)


def test_safe_select_sql_accepts_read_only_queries():
    assert is_safe_select_sql("SELECT * FROM orders")
    assert is_safe_select_sql("WITH recent AS (SELECT * FROM orders) SELECT * FROM recent")


def test_safe_select_sql_rejects_write_operations():
    assert not is_safe_select_sql("DELETE FROM orders")
    assert not is_safe_select_sql("SELECT * FROM orders; DROP TABLE users;")
    assert not is_safe_select_sql("CREATE TABLE mart AS SELECT * FROM orders")


def test_datamart_table_name_validation():
    assert is_safe_datamart_table_name("daily_sales_mart")
    assert not is_safe_datamart_table_name("analytics.daily_sales_mart")
    assert not is_safe_datamart_table_name("daily-sales")


def test_metadata_context_tolerates_missing_datamart_config(monkeypatch):
    monkeypatch.setattr(
        "sql_agent.sql_agent.load_all_metadata",
        lambda: {"schema_json": {"orders": {}}, "integrity_json": {"summary": {}}},
    )
    monkeypatch.setattr(
        "sql_agent.sql_agent._get_datamart_engine",
        lambda: (_ for _ in ()).throw(ValueError("datamart DB configuration is incomplete. Missing: password")),
    )

    payload = json.loads(get_metadata_context.invoke({}))

    assert payload["datamart_status"] == "unavailable"
    assert payload["datamart_schema"] == {}
    assert "Missing: password" in payload["datamart_error"]
