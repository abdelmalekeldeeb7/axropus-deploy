Ticket Triage Workflow (End-to-End)

Steps
1) Submit job (subject/body/customer/channel)
2) Router normalizes to canonical schema
3) Runtime executes deterministic inference
4) AMF replay applies if eligible
5) Output: priority, summary, tags, assignee
6) Audit ledger records decisions + metrics + output
7) MF snapshot stored
8) Client retrieves status/logs/history

Expected Log Signals
- [AMF_HIT] or [AMF_MISS]
- [AMF_SKIP] with skip_ratio
- [KORITH_RUN_SUMMARY]
- [KORITH_HEALTH]
- [MF_RESTORE] on restore
