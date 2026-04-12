# Axropus AMF vs LMCache — H200 head-to-head

This is the customer-facing benchmark deliverable. Run it on a single
H200 node with one Llama-3.1-70B replica and a fixed 500-request
workload drawn from the shared-prefix agent chat pattern.

## Workload

```
n_requests          : 500
shared_prefix_count : 10
tokens_per_prefix   : 2048
suffix_tokens       : 64
model               : meta-llama/Llama-3.1-70B
gpu                 : 1 x H200 SXM (141 GB HBM3e)
```

## Configurations

| Variant        | KV backend                              | Storage            |
|----------------|-----------------------------------------|--------------------|
| vllm-cold      | vLLM default APC                        | FP16 paged         |
| lmcache-cpu    | vLLM + LMCache G2 (CPU DRAM)            | FP16 on CPU        |
| axropus-int4   | Axropus AMF G1 + LMCache G2 fallback     | INT4 per-block     |
| axropus-tq     | Axropus AMF G1 + TurboQuant + LMCache   | sub-4-bit TurboQuant |

## Methodology

1. Warm up each variant with 50 unique prefixes to prime all caches.
2. Measure wall-clock latency for the next 500 mixed requests.
3. Record TTFT (time-to-first-token), total decode latency, and tokens
   per second across the run.
4. Export Prometheus metrics throughout and attach Grafana snapshots at
   ``deploy/observability/axropus_grafana.json``.

## Reproducing

```bash
# Build and install
pip install -e '.[server,lmcache,bench]'

# Run the full sweep (takes ~10 minutes on a single H200)
axropus bench --config benchmarks/standard.yaml --output results.json

# Or drive the simulation-only harness (no GPU required)
python -m benchmarks.multi_request_benchmark
```

## Expected results (placeholder — fill in after the first H200 run)

| Variant       | p50 TTFT (ms) | p95 TTFT (ms) | Throughput (tok/s) | Hit rate |
|---------------|---------------|---------------|--------------------|----------|
| vllm-cold     | TBD           | TBD           | TBD                | 0.00     |
| lmcache-cpu   | TBD           | TBD           | TBD                | TBD      |
| axropus-int4  | TBD           | TBD           | TBD                | TBD      |
| axropus-tq    | TBD           | TBD           | TBD                | TBD      |

The number that matters most for the customer briefing is the
AMF G1 lookup latency — expected to be well under 1 ms on H200 thanks
to zero PCIe traversal on warm hits — compared to 50-300 ms for LMCache
CPU and 1-5 s for LMCache NVMe.
