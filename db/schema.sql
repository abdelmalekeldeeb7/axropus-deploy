PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  company_name TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_keys (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  key TEXT UNIQUE NOT NULL,
  status TEXT DEFAULT 'trial',
  tier TEXT DEFAULT 'trial',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS deployments (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  api_key_id INTEGER,
  runtime TEXT NOT NULL,
  model_family TEXT NOT NULL,
  model_size TEXT NOT NULL,
  draft_model TEXT,
  status TEXT DEFAULT 'pending',
  deployed_at TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id),
  FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY,
  api_key_id INTEGER NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  tokens_processed INTEGER DEFAULT 0,
  interval_seconds INTEGER DEFAULT 60,
  prefix_skipped INTEGER DEFAULT 0,
  decode_accelerated INTEGER DEFAULT 0,
  amf_hit_rate REAL DEFAULT 0,
  spec_acceptance_rate REAL DEFAULT 0,
  effective_tps REAL DEFAULT 0,
  baseline_tps REAL DEFAULT 0,
  compute_saved_pct REAL DEFAULT 0,
  gpu_count INTEGER DEFAULT 0,
  model_family TEXT,
  model_size_bucket TEXT,
  adapter_type TEXT,
  sdk_version TEXT,
  license_id TEXT,
  heartbeat INTEGER DEFAULT 0,
  FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
);

CREATE TABLE IF NOT EXISTS invoices (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  period_start DATE,
  period_end DATE,
  total_tokens INTEGER,
  amount_cents INTEGER,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);
