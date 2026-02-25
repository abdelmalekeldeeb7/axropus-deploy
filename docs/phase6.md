Phase 6: Kernel + Speculative Decode Acceleration

Overview
- Adds capability-gated accel/spec execution modes to the existing adapter/worker pipeline.
- Keeps baseline path as default and fail-safe.
- Integrates spec and engine fields into authoritative `metrics.json`.

Execution Modes
- `baseline`:
  - Calls `adapter.run_baseline(...)`.
  - No acceleration, no speculative decode.
- `accel`:
  - Calls `adapter.run_accel(...)` when `KORITH_ACCEL_ENABLED=1`.
  - Emits `engine.mode=accel`.
- `speculative`:
  - Calls `adapter.run_speculative(...)` only when all are true:
    - `policy.allow_spec=true`
    - `KORITH_SPEC_ENABLED=1`
    - adapter capability: `DRAFT_SUPPORTED=true`
    - adapter capability: `VERIFY_TOKENS=true` or `LOGITS_ACCESS=true`
  - Emits `SPEC_ENABLE` and `SPEC_SUMMARY` events.
  - Auto-disables temporarily on low acceptance EMA.

Backend Support
- `korith_local`:
  - Baseline + accel entrypoints.
  - Spec capability disabled by default.
- `korith_cuda`:
  - New backend ID.
  - Capability flags expose `VERIFY_TOKENS`, `LOGITS_ACCESS`, `DRAFT_SUPPORTED`.
  - Uses same runtime binary path, with accel/spec env toggles.
- `openai_compatible`, `vllm`, `hf_transformers`:
  - Baseline only; spec disabled with explicit reason in metrics.

Snapshot Safety
- Snapshot metadata sidecar: `<mf_snapshot>.meta.json`.
- Metadata includes:
  - `fingerprint_hash`, `model_hash`, `tokenizer_hash`, `n_ctx`
  - `kv_layout_version`
  - `checksum_sha256`
- On restore, mismatch emits `CORRUPTION_DETECTED` and replay is blocked (fail closed).

Config Knobs
- `KORITH_ACCEL_ENABLED` (default `0`)
- `KORITH_CUDA_DEVICE` (default worker gpu id)
- `KORITH_CUDA_DTYPE` (default `fp16`)
- `KORITH_KV_LAYOUT_VERSION` (default `v1`)
- `KORITH_SPEC_ENABLED` (default `0`)
- `KORITH_SPEC_K` (default `6`)
- `KORITH_SPEC_MIN_ACCEPT` (default `0.55`)
- `KORITH_SPEC_DISABLE_AFTER_N` (default `50`)

Build Commands
- Core platform tests:
  - `python3 -m unittest discover -s platform/tests -v`
- Existing engine binary:
  - `cmake -S . -B build`
  - `cmake --build build -j"$(nproc)"`
- Optional Phase 6 engine shared-library stub:
  - `cmake -S platform/engine/cpp -B build/engine`
  - `cmake --build build/engine -j"$(nproc)"`
  - outputs `build/engine/libkorith_engine.so`

Run Commands
- Start worker/router with accel enabled:
  - `KORITH_ACCEL_ENABLED=1 KORITH_CUDA_DEVICE=0 python3 platform/korith_platform.py worker --worker-id worker-gpu0 --gpu-id 0 --host 127.0.0.1 --port 9000`
  - `python3 platform/korith_platform.py router --host 127.0.0.1 --port 8000`
- Enable spec:
  - `KORITH_SPEC_ENABLED=1 KORITH_SPEC_K=6 KORITH_SPEC_MIN_ACCEPT=0.55 KORITH_SPEC_DISABLE_AFTER_N=50 ...`

Bench Command
- `python3 platform/korith_platform.py bench spec --jobspec demo/jobs/ticket_triage.json --runs 10 --url http://127.0.0.1:8000 --api-key "$KORITH_API_KEY"`
- `./scripts/spec_jobs_check.sh demo/jobs/ticket_triage.json 5 60 6`
- `./scripts/phase6_golden_benchmark.sh demo/jobs/ticket_triage.json 3 60`

Status Endpoints and CLI
- HTTP:
  - `GET /v1/spec/status?fingerprint=<fingerprint_hash>`
  - `GET /v1/kernels/status`
- CLI:
  - `python3 platform/korith_platform.py spec status <fingerprint_hash> --url http://127.0.0.1:8000 --api-key "$KORITH_API_KEY"`
  - `python3 platform/korith_platform.py kernels status --url http://127.0.0.1:8000 --api-key "$KORITH_API_KEY"`

Metrics Interpretation
- `spec.acceptance_rate`:
  - Fraction of proposed draft tokens accepted by verify path.
  - Low sustained values trigger temporary spec disable.
- `spec.speedup_est`:
  - Adapter-reported estimate for relative speedup (mode-dependent).
- `amf.roi`:
  - Prefix replay value; still authoritative for replay economics.

Limitations (current phase)
- CUDA engine shared library is a stable interface + stub; runtime remains compatible without CUDA.
- `korith_cuda` currently routes through existing runtime binary path with accel/spec toggles.
- Deep fused kernels and multi-model draft/verify optimization are left for next iterations.
