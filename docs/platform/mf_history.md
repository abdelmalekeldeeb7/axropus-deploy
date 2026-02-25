MF History Restore Design

Snapshot Fields
- mf_snapshot_id
- job_id
- created_at
- min_admit_roi
- eviction_pressure
- replay_disable_mask
- cooldown_ms
- model_hash
- prompt_hash

Rules
- Restore only if model_hash + prompt_hash match current job
- Restore must not override AMF hard disable
- Restore must log:
  [MF_RESTORE] job_id=... min_admit_roi=... evict_pressure=... replay_mask=0x.. cooldown_ms=...
- Restore is deterministic and idempotent for same snapshot
