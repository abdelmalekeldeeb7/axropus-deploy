Pilot Success Criteria

Primary KPIs
1. Prefill skip efficiency
- `avg_skip_ratio >= 0.30` on target repeated workloads

2. Replay return
- `avg_roi >= 1.0` after baseline stabilization

3. Cost reduction
- `cost_saved_usd / cost_total_usd >= 0.15` on replay-eligible traffic

4. Latency
- P50 or P95 total latency improvement on repeated prompt cohorts

Safety KPIs
1. Replay fail-closed behavior
- Restore mismatch or corruption always falls back to baseline without crash

2. Governance stability
- No uncontrolled replay oscillation
- Negative ROI streak triggers cooldown behavior

Evidence Artifacts
- `metrics.json`
- `events.jsonl`
- `platform_data/savings_report.json`
- `platform_data/executive_summary.pdf`
