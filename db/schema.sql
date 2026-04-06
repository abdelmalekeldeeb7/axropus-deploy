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

-- ═══════════════════════════════════════════════════════════
-- AXROPUS PLATFORM HUB TABLES
-- ═══════════════════════════════════════════════════════════

-- Model deployments (managed by model_manager)
CREATE TABLE IF NOT EXISTS model_deployments (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_family TEXT NOT NULL,
    status TEXT DEFAULT 'stopped',
    gpu_id TEXT,
    quant_mode TEXT DEFAULT 'int4',
    vram_allocated_gb REAL,
    amf_enabled INTEGER DEFAULT 1,
    config JSON,
    pid INTEGER,
    port INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OpenClaw agents (Claws)
CREATE TABLE IF NOT EXISTS claws (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    system_prompt TEXT,
    tools JSON,
    channels JSON,
    openclaw_config JSON,
    amf_config JSON,
    status TEXT DEFAULT 'draft',
    total_tasks INTEGER DEFAULT 0,
    total_tokens_saved INTEGER DEFAULT 0,
    avg_prefix_reuse REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claw task executions
CREATE TABLE IF NOT EXISTS claw_tasks (
    id TEXT PRIMARY KEY,
    claw_id TEXT NOT NULL REFERENCES claws(id),
    input TEXT,
    output TEXT,
    total_steps INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    prefix_reuse_rate REAL DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual steps within a claw task
CREATE TABLE IF NOT EXISTS claw_steps (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES claw_tasks(id),
    step_number INTEGER NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    amf_hit INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    restore_ms REAL DEFAULT 0,
    prefill_ms REAL DEFAULT 0,
    decode_ms REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Economics / savings tracking (daily aggregates)
CREATE TABLE IF NOT EXISTS economics_daily (
    date TEXT NOT NULL,
    model_id TEXT DEFAULT '__all__',
    claw_id TEXT DEFAULT '__all__',
    total_tokens INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    compute_cost_usd REAL DEFAULT 0,
    equivalent_openai_usd REAL DEFAULT 0,
    equivalent_anthropic_usd REAL DEFAULT 0,
    equivalent_together_usd REAL DEFAULT 0,
    amf_hit_rate REAL DEFAULT 0,
    avg_prefix_reuse REAL DEFAULT 0,
    total_requests INTEGER DEFAULT 0,
    PRIMARY KEY (date, model_id, claw_id)
);

-- Request log for billing and analytics
CREATE TABLE IF NOT EXISTS request_log (
    id TEXT PRIMARY KEY,
    api_key_id TEXT,
    model_id TEXT,
    claw_id TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    amf_hit INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    restore_ms REAL DEFAULT 0,
    total_ms REAL DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_claw_tasks_claw ON claw_tasks(claw_id);
CREATE INDEX IF NOT EXISTS idx_claw_steps_task ON claw_steps(task_id);
CREATE INDEX IF NOT EXISTS idx_economics_date ON economics_daily(date);
CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at);
CREATE INDEX IF NOT EXISTS idx_model_deploy_status ON model_deployments(status);
