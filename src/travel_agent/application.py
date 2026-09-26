from pathlib import Path

from .repository import SQLiteRepository
from .service import ConsultationService


def checkpoint_database_path(database_path: str | Path) -> Path:
    database_path = Path(database_path)
    return database_path.with_name(f"{database_path.stem}_checkpoints.sqlite")


def create_consultation_service(
    database_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> ConsultationService:
    """Build the application service used by every entry point."""
    return ConsultationService(
        SQLiteRepository(database_path),
        checkpoint_path=checkpoint_path or checkpoint_database_path(database_path),
    )
