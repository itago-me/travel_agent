import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain import AuditEvent, Consultation, Message, utc_now


class SQLiteRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS consultations (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    consultant_id TEXT NOT NULL,
                    customer_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trip_requirements TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                """
            )

    def create_consultation(self, consultation: Consultation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO consultations
                    (id, thread_id, consultant_id, customer_name, status,
                     trip_requirements, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    consultation.id,
                    consultation.thread_id,
                    consultation.consultant_id,
                    consultation.customer_name,
                    consultation.status,
                    json.dumps(consultation.trip_requirements, ensure_ascii=False),
                    consultation.created_at,
                    consultation.updated_at,
                ),
            )
            for message in consultation.messages:
                self._insert_message(connection, consultation.id, message)

    def get_consultation(self, consultation_id: str) -> Consultation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM consultations WHERE id = ?", (consultation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Consultation not found: {consultation_id}")
            messages = connection.execute(
                "SELECT role, content FROM messages WHERE consultation_id = ? ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return Consultation(
            id=row["id"],
            thread_id=row["thread_id"],
            consultant_id=row["consultant_id"],
            customer_name=row["customer_name"],
            status=row["status"],
            messages=[Message(item["role"], item["content"]) for item in messages],
            trip_requirements=json.loads(row["trip_requirements"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def append_message(self, consultation_id: str, message: Message) -> None:
        now = utc_now()
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            self._insert_message(connection, consultation_id, message, now)
            connection.execute(
                "UPDATE consultations SET updated_at = ? WHERE id = ?",
                (now, consultation_id),
            )

    def update_requirements(self, consultation_id: str, requirements: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET trip_requirements = ?, updated_at = ? WHERE id = ?",
                (json.dumps(requirements, ensure_ascii=False), utc_now(), consultation_id),
            )

    def add_audit_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            self._ensure_exists(connection, event.consultation_id)
            connection.execute(
                "INSERT INTO audit_events (id, consultation_id, event_type, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    event.consultation_id,
                    event.event_type,
                    json.dumps(event.details, ensure_ascii=False),
                    event.created_at,
                ),
            )

    def list_audit_events(self, consultation_id: str) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE consultation_id = ? ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return [
            AuditEvent(
                consultation_id=row["consultation_id"],
                event_type=row["event_type"],
                details=json.loads(row["details"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _insert_message(
        connection: sqlite3.Connection,
        consultation_id: str,
        message: Message,
        created_at: str | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO messages (id, consultation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), consultation_id, message.role, message.content, created_at or utc_now()),
        )

    @staticmethod
    def _ensure_exists(connection: sqlite3.Connection, consultation_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM consultations WHERE id = ?", (consultation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Consultation not found: {consultation_id}")
