Workload Requirements for Pilot

Ideal Workload Shapes
1. Repeated prefix templates
- Support ticket triage
- RAG prompts with fixed system/context headers
- Code review prompts with stable rubric

2. Near-miss prompts
- Same prefix, variable suffix
- Used to validate lane assignment and replay miss handling

3. Long output prompts
- Needed to measure decode throughput and queue effects

4. Mixed sessions
- Repeated prompts from same `session_id`
- Verifies session affinity and HIT lane behavior

Data Requirements
- 100-1000 representative prompts
- At least 30% expected prefix reuse
- Deterministic config pinned (`seed`, `n_ctx`, `n_batch`, sampling)
