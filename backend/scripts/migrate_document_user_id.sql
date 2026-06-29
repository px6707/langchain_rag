-- Document user_id migration (run against existing PostgreSQL DB)
-- Assigns legacy documents to the first admin user, then enforces NOT NULL + FK.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS user_id UUID NULL;

UPDATE documents d
SET user_id = (
    SELECT u.id FROM users u WHERE u.is_admin = true ORDER BY u.created_at ASC LIMIT 1
)
WHERE d.user_id IS NULL;

-- Fail fast if no admin exists to own legacy rows
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM documents WHERE user_id IS NULL) THEN
        RAISE EXCEPTION 'Cannot backfill documents.user_id: no admin user found';
    END IF;
END $$;

ALTER TABLE documents
    ALTER COLUMN user_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_documents_user_id'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT fk_documents_user_id
            FOREIGN KEY (user_id) REFERENCES users(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_documents_user_id ON documents(user_id);
