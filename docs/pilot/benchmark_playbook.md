Benchmark Playbook

Purpose
- Produce repeatable replay and economics evidence for pilot decision makers.

Test Matrix
1. Cold/Warm/Warm
- Run identical workload three times.
- Expect first miss then replay hits for replay-capable backends.

2. Near-Miss
- Keep prefix constant and change short suffix.
- Measure partial reuse and lane behavior.

3. Long Output
- Use longer `max_tokens` to evaluate decode throughput and queue pressure.

Execution
1. Start platform:
- `bash deploy/install.sh`
2. Submit workload set:
- `python platform/korith_platform.py submit demo/jobs/ticket_triage.json --url http://127.0.0.1:8080`
3. Collect telemetry:
- `python platform/korith_platform.py metrics <job_id> --url http://127.0.0.1:8080`
- `python platform/korith_platform.py events <job_id> --url http://127.0.0.1:8080`
4. Generate report:
- `python platform/korith_platform.py report --ledger ./platform_data/ledger.sqlite --out ./platform_data/savings_report.json --gpu_hourly_cost 2.5`

Outputs
- `savings_report.json`
- `savings_report.csv`
- `executive_summary.pdf`
