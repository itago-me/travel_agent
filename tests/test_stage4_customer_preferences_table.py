import sqlite3

import pytest

from travel_agent.repository import SQLiteRepository


def _connect(database):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    return connection


def test_repository_creates_customer_preferences_table_with_expected_contract(tmp_path):
    database = tmp_path / "travel.db"

    SQLiteRepository(database)

    with _connect(database) as connection:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(customer_preferences)")
        }
        indexes = connection.execute(
            "PRAGMA index_list(customer_preferences)"
        ).fetchall()

    assert set(columns) == {
        "customer_id",
        "preference_key",
        "preference_value",
        "source",
        "created_at",
        "updated_at",
    }
    assert columns["customer_id"]["notnull"] == 1
    assert columns["preference_key"]["notnull"] == 1
    assert columns["preference_value"]["notnull"] == 1
    assert any(row["unique"] == 1 for row in indexes)


def test_customer_preference_identity_is_scoped_by_customer(tmp_path):
    database = tmp_path / "travel.db"
    SQLiteRepository(database)
    insert_sql = """
        INSERT INTO customer_preferences (
            customer_id, preference_key, preference_value,
            source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    timestamp = "2026-10-10T00:00:00+00:00"

    with _connect(database) as connection:
        connection.execute(
            insert_sql,
            (
                "customer-001",
                "hotel.star_level",
                '"舒适型"',
                "explicit",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            insert_sql,
            (
                "customer-002",
                "hotel.star_level",
                '"豪华型"',
                "explicit",
                timestamp,
                timestamp,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                insert_sql,
                (
                    "customer-001",
                    "hotel.star_level",
                    '"经济型"',
                    "explicit",
                    timestamp,
                    timestamp,
                ),
            )


def test_repository_adds_preferences_table_to_existing_database(tmp_path):
    database = tmp_path / "travel.db"
    with _connect(database) as connection:
        connection.execute("CREATE TABLE legacy_data (id TEXT PRIMARY KEY)")

    SQLiteRepository(database)

    with _connect(database) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("customer_preferences",),
        ).fetchone()

    assert table["name"] == "customer_preferences"
