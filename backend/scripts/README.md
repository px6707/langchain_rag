# Deprecated SQL migrations

Database schema is managed by Alembic (`backend/alembic/versions/`).

The former hand-written SQL files in this directory have been removed. Use:

```bash
cd backend
alembic upgrade head      # apply migrations
alembic stamp head        # mark current DB as up-to-date without running DDL
```
