from unittest.mock import MagicMock, patch

from app.db.migrate import run_migrations


def test_run_migrations_calls_alembic_upgrade():
    with patch("app.db.migrate.command.upgrade") as mock_upgrade:
        with patch("app.db.migrate.settings") as mock_settings:
            mock_settings.auto_db_migrate = True
            run_migrations()
            mock_upgrade.assert_called_once()
            cfg = mock_upgrade.call_args.args[0]
            assert cfg.get_main_option("script_location") is not None


def test_run_migrations_skipped_when_disabled():
    with patch("app.db.migrate.command.upgrade") as mock_upgrade:
        with patch("app.db.migrate.settings") as mock_settings:
            mock_settings.auto_db_migrate = False
            run_migrations()
            mock_upgrade.assert_not_called()


def test_run_migrations_idempotent_upgrade():
    with patch("app.db.migrate.command.upgrade") as mock_upgrade:
        with patch("app.db.migrate.settings") as mock_settings:
            mock_settings.auto_db_migrate = True
            run_migrations()
            run_migrations()
            assert mock_upgrade.call_count == 2
