import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.config import settings

logger = logging.getLogger(__name__)

_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def run_migrations() -> None:
    if not settings.auto_db_migrate:
        logger.info("Database auto-migration disabled (AUTO_DB_MIGRATE=false)")
        return
    cfg = Config(str(_ALEMBIC_INI))
    logger.info("Running Alembic migrations to head")
    command.upgrade(cfg, "head")
