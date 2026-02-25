#pragma once

// Minimal, stable engine trait interface for uniform scheduling.
//
// Design goals:
// - C++ compatible, no templates in the public surface.
// - Plain structs for ABI/FFI-safe data exchange.
// - Opaque `Context` type so engines can share a common call signature without
//   coupling this header to a specific runtime or backend.

namespace korith::core {

// Execution context passed to engines.
//
// This remains minimal and stable; callers can stash opaque state as needed.
struct Context {
  void * opaque = nullptr;
  bool baseline_ready = false;
};

// Scalar signals used by a scheduler to compare candidate engines.
//
// All fields are plain primitives for ABI stability.
struct EngineSignal {
  float confidence;
  float cost_estimate;
  float benefit_estimate;
  int suggested_depth;
};

// Base class for schedulable engines.
//
// NOTE: This is a C++ vtable-based interface (not a C ABI). Only `EngineSignal`
// is intended to be trivially passable across FFI boundaries.
class KorithEngine {
public:
  virtual ~KorithEngine() = default;

  // Returns the engine's current scheduling signal for the provided context.
  virtual EngineSignal evaluate(Context & ctx) = 0;

  // Executes the engine's scheduled work for the provided context.
  virtual void execute(Context & ctx) = 0;
};

}  // namespace korith::core
