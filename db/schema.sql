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
  inference_url TEXT,
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

-- ═══════════════════════════════════════════════════════════════════════════
-- Axropus Platform Hub — Model Deployments
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS model_deployments (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  status TEXT DEFAULT 'pending',
  port INTEGER,
  quant_mode TEXT,
  tensor_parallel INTEGER DEFAULT 1,
  gpu_count INTEGER DEFAULT 0,
  error_message TEXT,
  deployed_at TIMESTAMP,
  stopped_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE INDEX IF NOT EXISTS idx_model_deployments_customer ON model_deployments(customer_id);
CREATE INDEX IF NOT EXISTS idx_model_deployments_status ON model_deployments(status);

-- ═══════════════════════════════════════════════════════════════════════════
-- Axropus Platform Hub — OpenClaw Agents (Claws)
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS claws (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  system_prompt TEXT DEFAULT 'You are a helpful AI assistant.',
  tools TEXT DEFAULT '[]',
  channels TEXT DEFAULT '[]',
  openclaw_config TEXT DEFAULT '{}',
  amf_config TEXT DEFAULT '{}',
  status TEXT DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE INDEX IF NOT EXISTS idx_claws_customer ON claws(customer_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- Axropus Platform Hub — Claw Task Executions
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS claw_tasks (
  id INTEGER PRIMARY KEY,
  claw_id INTEGER NOT NULL,
  task_uid TEXT UNIQUE NOT NULL,
  prompt TEXT NOT NULL,
  status TEXT DEFAULT 'pending',
  result TEXT,
  total_steps INTEGER DEFAULT 0,
  tokens_used INTEGER DEFAULT 0,
  tokens_saved INTEGER DEFAULT 0,
  duration_ms INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  FOREIGN KEY (claw_id) REFERENCES claws(id)
);

CREATE INDEX IF NOT EXISTS idx_claw_tasks_claw ON claw_tasks(claw_id);
CREATE INDEX IF NOT EXISTS idx_claw_tasks_uid ON claw_tasks(task_uid);

-- ═══════════════════════════════════════════════════════════════════════════
-- Axropus Platform Hub — Individual Steps Within a Claw Task
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS claw_steps (
  id INTEGER PRIMARY KEY,
  task_id INTEGER NOT NULL,
  step_number INTEGER NOT NULL,
  action TEXT,
  input_text TEXT,
  output_text TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  tokens_saved INTEGER DEFAULT 0,
  amf_hit INTEGER DEFAULT 0,
  duration_ms INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (task_id) REFERENCES claw_tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_claw_steps_task ON claw_steps(task_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- Axropus Platform Hub — Daily Economics / Savings Aggregates
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS economics_daily (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  date TEXT NOT NULL,
  total_tokens INTEGER DEFAULT 0,
  tokens_saved INTEGER DEFAULT 0,
  axropus_cost_usd REAL DEFAULT 0,
  openai_equivalent_usd REAL DEFAULT 0,
  anthropic_equivalent_usd REAL DEFAULT 0,
  together_equivalent_usd REAL DEFAULT 0,
  amf_hit_rate REAL DEFAULT 0,
  prefix_reuse_rate REAL DEFAULT 0,
  total_requests INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_economics_daily_customer_date ON economics_daily(customer_id, date);
