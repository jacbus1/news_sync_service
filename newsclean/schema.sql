CREATE TABLE IF NOT EXISTS events_raw (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  title TEXT,
  url TEXT,
  published_at TEXT,
  ingested_at TEXT NOT NULL,
  content TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_url ON events_raw(url);

-- Track which individual items were already posted to Telegram to prevent re-posting.
CREATE TABLE IF NOT EXISTS sent_items (
  key TEXT PRIMARY KEY,
  url TEXT,
  title TEXT,
  first_seen_at TEXT NOT NULL,
  sent_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sent_items_sent_at ON sent_items(sent_at);

CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  item_count INTEGER NOT NULL,
  summary_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  report_date TEXT NOT NULL,
  fgi_value INTEGER NOT NULL,
  report_text TEXT NOT NULL
);

-- Draft/approved daily digest that is reviewed before posting to the public channel.
CREATE TABLE IF NOT EXISTS daily_digests (
  digest_date_et TEXT PRIMARY KEY,
  status TEXT NOT NULL, -- draft|sent
  created_at TEXT NOT NULL,
  draft_text TEXT NOT NULL,
  draft_chat_id TEXT,
  draft_message_id INTEGER,
  sent_at TEXT
);

-- Generic key/value state (e.g., Telegram update offset).
CREATE TABLE IF NOT EXISTS bot_state (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
