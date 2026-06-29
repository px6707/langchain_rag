-- Parse job lease / generation migration (run against existing PostgreSQL DB)
-- Required when upgrading from versions without lease fields.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS active_parse_generation INTEGER NOT NULL DEFAULT 1;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS active_job_id UUID NULL;

ALTER TABLE parse_jobs
    ADD COLUMN IF NOT EXISTS parse_generation INTEGER NOT NULL DEFAULT 1;

ALTER TABLE parse_jobs
    ADD COLUMN IF NOT EXISTS lease_token UUID NULL;

ALTER TABLE parse_jobs
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ NULL;

ALTER TABLE parse_jobs
    ADD COLUMN IF NOT EXISTS worker_id VARCHAR(128) NULL;

UPDATE documents SET active_parse_generation = 1 WHERE active_parse_generation IS NULL;
UPDATE parse_jobs SET parse_generation = 1 WHERE parse_generation IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_parse_jobs_document_running
    ON parse_jobs (document_id) WHERE status = 'running';
