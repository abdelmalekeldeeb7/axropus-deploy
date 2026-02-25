# Korith

Korith is a minimal, production-grade architecture for **dynamic AI inference control**:

- **C++** is a thin, stable inference core that embeds `llama.cpp` and exposes a **C ABI**.
- **Python** is the dynamic control plane (policy, scheduling, power feedback) and owns the **token generation loop**.

This split keeps the high-frequency policy iteration in Python while minimizing the surface area exposed to `llama.cpp` API churn.

## Why control is in Python

Control policies evolve quickly: power caps, multi-stream scheduling, admission control, and per-request QoS. Python enables rapid iteration, experimentation, and integration with system telemetry without recompiling the inference core.

## Why C++ is minimal

The C++ layer is only responsible for:

- Loading the GGUF model (`llama_model_load_from_file`)
- Creating a context (`llama_init_from_model`)
- Evaluating a prompt (prefill) using `llama_batch`
- Producing **one token per call** via `korith_step()` (greedy sampling)

No scheduling, no control loop, no printing, and no threading logic lives in C++.

## Build (libkorith.so)

Prereqs:

- GCC 13+
- A CUDA-enabled `llama.cpp` build producing:
  - `$LLAMA_ROOT/include/llama.h`
  - `$LLAMA_ROOT/build/bin/libllama.so`

Build the C API shared library into `./build`:

```bash
export LLAMA_ROOT=/path/to/llama.cpp
cmake -S korith_c_api -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/libkorith.so` with an RPATH pointing at `$LLAMA_ROOT/build/bin` so it can resolve `libllama.so` at runtime.

## Run (Python control plane)

Prereqs:

- Python 3.10+
- `pynvml` installed (`pip install pynvml`)

Run the controller:

```bash
python3 python/controller.py --model /path/to/model.gguf --prompt "Hello from Korith" --max-tokens 256 --n-ctx 4096 --gpu 0
```

The controller:

- Calls `korith_init()`
- Calls `korith_tokenize()` for prefill
- Loops calling `korith_step()` token-by-token
- Samples GPU power via NVML and applies a simple rule-based policy

## Future scheduling

This architecture cleanly enables:

- Multi-stream decode scheduling in Python (multiple contexts, per-stream budgets)
- Tokens-per-watt optimization under power caps
- OS-level telemetry feedback (power, clocks, thermals, utilization)
- Agentic policies that adapt at token-step granularity

