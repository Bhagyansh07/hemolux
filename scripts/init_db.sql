-- Hemolux telemetry schema (Cloudflare D1 / SQLite).
--
-- Two tables, no foreign keys, no user table. created_at is written from the
-- request arrival time by the worker; the client never supplies it, so a row
-- cannot be backdated or forged. There is no column that could hold an
-- identifier, an image, or free text, by construction rather than by policy.
--
-- Apply with:
--   wrangler d1 execute hemolux-telemetry --remote --file scripts/init_db.sql

CREATE TABLE IF NOT EXISTS telemetry (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at  TEXT    NOT NULL,
  model_id    TEXT    NOT NULL,
  band        TEXT    NOT NULL,
  hb_hat      REAL,
  sigma       REAL,
  quality     REAL,
  latency_ms  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_telemetry_created ON telemetry (created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_band    ON telemetry (band);

CREATE TABLE IF NOT EXISTS feedback (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at  TEXT NOT NULL,
  hb          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at);
