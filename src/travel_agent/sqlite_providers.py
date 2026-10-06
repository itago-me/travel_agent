"""SQLite-backed provider implementations used by the real Agent tools.

The providers implement the same contracts as external suppliers. SQLite is
only the first data source; the Agent sees the normalized provider schemas.
"""

import sqlite3
from datetime import date, time
from pathlib import Path

from .providers import (
    AttractionSearchRequest,
    AttractionSearchResult,
    AttractionOption,
    HotelSearchRequest,
    HotelSearchResult,
    HotelOption,
    TransportMode,
    TransportSearchRequest,
    TransportSearchResult,
    TransportOption,
)


def _connect(database: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_travel_data(database: str | Path) -> None:
    """Create and seed the small local catalog used by the Agent demo."""
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with _connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS travel_transport_options (
                option_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                departure_date TEXT NOT NULL,
                departure_time TEXT NOT NULL,
                arrival_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                total_price REAL NOT NULL,
                provider TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS travel_hotel_options (
                option_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                price_per_night REAL NOT NULL,
                rating REAL NOT NULL,
                available_rooms INTEGER NOT NULL,
                provider TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS travel_attraction_options (
                option_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                ticket_price REAL NOT NULL,
                provider TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO travel_transport_options
                (option_id, mode, origin, destination, departure_date,
                 departure_time, arrival_time, duration_minutes, total_price, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("train-bj-hz-001", "train", "北京", "杭州", "2026-10-01", "08:00", "13:30", 330, 1800.0, "sqlite_transport"),
                ("train-bj-hz-002", "train", "北京", "杭州", "2026-10-01", "14:00", "19:45", 345, 1500.0, "sqlite_transport"),
            ],
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO travel_hotel_options
                (option_id, name, city, price_per_night, rating, available_rooms, provider)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("hotel-hz-001", "西湖假日酒店", "杭州", 680.0, 4.6, 5, "sqlite_hotel"),
                ("hotel-hz-002", "钱塘精选酒店", "杭州", 420.0, 4.1, 8, "sqlite_hotel"),
            ],
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO travel_attraction_options
                (option_id, name, city, duration_minutes, ticket_price, provider)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("attraction-hz-001", "西湖", "杭州", 180, 0.0, "sqlite_attraction"),
                ("attraction-hz-002", "灵隐寺", "杭州", 150, 75.0, "sqlite_attraction"),
            ],
        )


class SQLiteTransportProvider:
    provider_name = "sqlite_transport"

    def __init__(self, database: str | Path):
        self.database = Path(database)
        initialize_travel_data(self.database)

    async def search(self, request: TransportSearchRequest) -> TransportSearchResult:
        query = """
            SELECT * FROM travel_transport_options
            WHERE origin = ? AND destination = ? AND departure_date = ?
        """
        params: list[object] = [
            request.origin,
            request.destination,
            request.departure_date.isoformat(),
        ]
        if request.mode is not None:
            query += " AND mode = ?"
            params.append(request.mode.value)
        query += " ORDER BY departure_time, option_id"
        with _connect(self.database) as connection:
            rows = connection.execute(query, params).fetchall()
        options = [
            TransportOption(
                provider=row["provider"],
                option_id=row["option_id"],
                mode=TransportMode(row["mode"]),
                origin=row["origin"],
                destination=row["destination"],
                departure_date=date.fromisoformat(row["departure_date"]),
                departure_time=time.fromisoformat(row["departure_time"]),
                arrival_time=time.fromisoformat(row["arrival_time"]),
                duration_minutes=row["duration_minutes"],
                total_price=row["total_price"] * request.traveler_count / 2,
            )
            for row in rows
        ]
        return TransportSearchResult(provider=self.provider_name, options=options)


class SQLiteHotelProvider:
    provider_name = "sqlite_hotel"

    def __init__(self, database: str | Path):
        self.database = Path(database)
        initialize_travel_data(self.database)

    async def search(self, request: HotelSearchRequest) -> HotelSearchResult:
        with _connect(self.database) as connection:
            rows = connection.execute(
                "SELECT * FROM travel_hotel_options WHERE city = ? ORDER BY price_per_night, option_id",
                (request.city,),
            ).fetchall()
        options = [
            HotelOption(
                provider=row["provider"],
                option_id=row["option_id"],
                name=row["name"],
                city=row["city"],
                price_per_night=row["price_per_night"] * request.rooms,
                rating=row["rating"],
                available_rooms=row["available_rooms"],
            )
            for row in rows
            if row["available_rooms"] >= request.rooms
        ]
        return HotelSearchResult(provider=self.provider_name, options=options)


class SQLiteAttractionProvider:
    provider_name = "sqlite_attraction"

    def __init__(self, database: str | Path):
        self.database = Path(database)
        initialize_travel_data(self.database)

    async def search(self, request: AttractionSearchRequest) -> AttractionSearchResult:
        with _connect(self.database) as connection:
            rows = connection.execute(
                "SELECT * FROM travel_attraction_options WHERE city = ? ORDER BY option_id",
                (request.city,),
            ).fetchall()
        options = [
            AttractionOption(
                provider=row["provider"],
                option_id=row["option_id"],
                name=row["name"],
                city=row["city"],
                visit_date=request.visit_date,
                duration_minutes=row["duration_minutes"],
                ticket_price=row["ticket_price"],
            )
            for row in rows
        ]
        return AttractionSearchResult(provider=self.provider_name, options=options)
