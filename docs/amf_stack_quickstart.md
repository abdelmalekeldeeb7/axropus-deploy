# AMF Stack Quickstart

This is the operational path for running vLLM with Axropus AMF enabled.

The benchmark harness proves behavior. This stack wrapper is the path for
starting a server.

## 1. Create Config

```bash
cd ~/amf
cp configs/amf_stack.env.example configs/amf_stack.env
```

Edit:

```bash
AMF_MODEL="/path/to/model"
AMF_MAX_MODEL_LEN="18432"
AMF_KV_CACHE_MEMORY_GB="8"
KORITH_VRAM_POOL_GB="16"
KORITH_VRAM_POOL_QUANT="fp8"
```

For same-total-GPU-memory tests, split memory explicitly:

```text
APC baseline:  vLLM KV = 24 GB
AMF stack:     vLLM KV = 8 GB, AMF pool = 16 GB
```

## 2. Verify Imports And Patches

```bash
./scripts/verify_amf_stack.sh configs/amf_stack.env
```

This verifies:

- vLLM imports
- `korith_vllm_ext` patches vLLM EngineCore
- `amf_register_prefix` exists
- `amf_get_cached_block_ids` exists
- AMF worker extension imports
- LMCache import status

LMCache is optional. A missing LMCache import is not a failure unless the config
enables LMCache.

## 3. Optional AMF + LMCache Smoke

Use this before claiming the combined stack is working on a new box:

```bash
RUN_AMF_LMCACHE_SMOKE=1 \
SMOKE_MAX_MODEL_LEN=1024 \
SMOKE_KV_CACHE_MB=384 \
./scripts/verify_amf_stack.sh configs/amf_stack.env
```

The smoke does:

1. Load vLLM with LMCache connector and AMF worker extension.
2. Run a cold prompt.
3. Save KV through AMF.
4. Reset vLLM prefix cache.
5. Restore and register through AMF.
6. Re-run and verify output token IDs match.

## 4. Start OpenAI-Compatible vLLM Server

```bash
./scripts/start_vllm_amf_stack.sh configs/amf_stack.env
```

The script launches:

```text
python -m korith_vllm_ext.korith_vllm_server --serve ...
```

That wrapper calls `vllm serve` with:

- `--enable-prefix-caching`
- AMF worker extension
- Korith scheduler
- AMF environment variables
- optional LMCache transfer config

## 5. Test The Server

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$AMF_MODEL"'",
    "prompt": "System: cache test. Repeat this context.\\nUser: say ok",
    "max_tokens": 8,
    "temperature": 0
  }'
```

## LMCache Mode

After the smoke passes, enable LMCache:

```bash
AXROPUS_ENABLE_LMCACHE="1"
AXROPUS_LMCACHE_FALLBACK="1"
LMCACHE_LOCAL_CPU="true"
LMCACHE_MAX_LOCAL_CPU_SIZE="32"
LMCACHE_CHUNK_SIZE="256"
```

Conceptually:

```text
AMF hot GPU tier first
LMCache lower CPU/NVMe/remote tier second
cold prefill last
```

Do not claim AMF+LMCache production behavior on a machine until
`RUN_AMF_LMCACHE_SMOKE=1 ./scripts/verify_amf_stack.sh ...` passes there.
