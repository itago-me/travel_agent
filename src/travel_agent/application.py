from pathlib import Path

from .repository import SQLiteRepository
from .service import ConsultationService


def create_consultation_service(database_path: str | Path) -> ConsultationService:
    """Build the application service used by every entry point."""
    return ConsultationService(SQLiteRepository(database_path))
