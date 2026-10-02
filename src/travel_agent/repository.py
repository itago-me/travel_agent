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
                    planning_constraints TEXT NOT NULL DEFAULT '{}',
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
                CREATE TABLE IF NOT EXISTS itinerary_drafts (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    selected_version_id TEXT,
                    selected_at TEXT,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS itinerary_versions (
                    id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (draft_id, version_number),
                    FOREIGN KEY (draft_id) REFERENCES itinerary_drafts(id)
                );
                CREATE TABLE IF NOT EXISTS provider_requests (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    call_number INTEGER NOT NULL,
                    response_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id),
                    FOREIGN KEY (version_id) REFERENCES itinerary_versions(id)
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(consultations)")}
            if "planning_constraints" not in columns:
                connection.execute(
                    "ALTER TABLE consultations ADD COLUMN planning_constraints TEXT NOT NULL DEFAULT '{}'"
                )
            draft_columns = {row[1] for row in connection.execute("PRAGMA table_info(itinerary_drafts)")}
            if "selected_version_id" not in draft_columns:
                connection.execute("ALTER TABLE itinerary_drafts ADD COLUMN selected_version_id TEXT")
            if "selected_at" not in draft_columns:
                connection.execute("ALTER TABLE itinerary_drafts ADD COLUMN selected_at TEXT")

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
            planning_constraints=json.loads(row["planning_constraints"]),
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

    def update_planning_constraints(self, consultation_id: str, constraints: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET planning_constraints = ?, updated_at = ? WHERE id = ?",
                (json.dumps(constraints, ensure_ascii=False), utc_now(), consultation_id),
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

    def save_itinerary_version(
        self,
        consultation_id: str,
        payload: dict[str, Any],
        tool_history: list[str],
        observations: dict[str, dict[str, Any]],
    ) -> tuple[str, int]:
        now = utc_now()
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            draft = connection.execute(
                "SELECT id FROM itinerary_drafts WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                draft_id = str(uuid4())
                connection.execute(
                    "INSERT INTO itinerary_drafts (id, consultation_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (draft_id, consultation_id, now, now),
                )
            else:
                draft_id = draft["id"]
                connection.execute(
                    "UPDATE itinerary_drafts SET updated_at = ? WHERE id = ?",
                    (now, draft_id),
                )
            row = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) AS latest FROM itinerary_versions WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
            version_number = int(row["latest"]) + 1
            version_id = str(uuid4())
            connection.execute(
                "INSERT INTO itinerary_versions (id, draft_id, version_number, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (version_id, draft_id, version_number, json.dumps(payload, ensure_ascii=False), now),
            )
            for call_number, tool_name in enumerate(tool_history, start=1):
                connection.execute(
                    "INSERT INTO provider_requests (id, consultation_id, version_id, tool_name, call_number, response_payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid4()), consultation_id, version_id, tool_name,
                        call_number,
                        json.dumps(observations.get(tool_name, {}), ensure_ascii=False),
                        now,
                    ),
                )
        return version_id, version_number

    def list_itinerary_versions(self, consultation_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT v.id, v.version_number, v.payload, v.created_at
                   FROM itinerary_versions v JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = ? ORDER BY v.version_number""",
                (consultation_id,),
            ).fetchall()
        return [
            {"id": row["id"], "version_number": row["version_number"],
             "payload": json.loads(row["payload"]), "created_at": row["created_at"]}
            for row in rows
        ]

    def get_itinerary_version(self, consultation_id: str, version_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT v.id, v.version_number, v.payload, v.created_at
                   FROM itinerary_versions v JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = ? AND v.id = ?""",
                (consultation_id, version_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Itinerary version not found: {version_id}")
        return {"id": row["id"], "version_number": row["version_number"],
                "payload": json.loads(row["payload"]), "created_at": row["created_at"]}

    def select_itinerary_version(self, consultation_id: str, version_id: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            draft = connection.execute(
                "SELECT id FROM itinerary_drafts WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                raise KeyError(f"Itinerary draft not found: {consultation_id}")
            version = connection.execute(
                "SELECT 1 FROM itinerary_versions WHERE id = ? AND draft_id = ?",
                (version_id, draft["id"]),
            ).fetchone()
            if version is None:
                raise KeyError(f"Itinerary version not found: {version_id}")
            connection.execute(
                "UPDATE itinerary_drafts SET selected_version_id = ?, selected_at = ?, updated_at = ? WHERE id = ?",
                (version_id, now, now, draft["id"]),
            )

    def get_selected_itinerary(self, consultation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT selected_version_id, selected_at FROM itinerary_drafts WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
        if row is None or row["selected_version_id"] is None:
            return None
        selected = self.get_itinerary_version(consultation_id, row["selected_version_id"])
        selected["selected_at"] = row["selected_at"]
        selected["version_id"] = selected.pop("id")
        return selected

    def list_provider_requests(self, consultation_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM provider_requests WHERE consultation_id = ? ORDER BY created_at, call_number, id",
                (consultation_id,),
            ).fetchall()
        return [
            {"id": row["id"], "version_id": row["version_id"],
             "tool_name": row["tool_name"], "call_number": row["call_number"],
             "response_payload": json.loads(row["response_payload"]),
             "created_at": row["created_at"]}
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
