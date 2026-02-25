//! C ABI for the authoritative Rust scheduler.
//!
//! This crate is built as a `cdylib` and intended to be loaded from C/C++.
//! The exported API includes `korith_schedule_step` and `korith_scheduler_update`.
//!
//! ## ABI / layout
//! All structs are `#[repr(C)]` and contain only POD fields (no pointers that
//! require ownership transfer, no heap allocation across the FFI boundary).
//!
//! The scheduler is deterministic; any per-loop state is owned by the caller and
//! passed via `KorithSchedulerState`.

use std::os::raw::c_int;
use std::sync::atomic::{AtomicU32, Ordering};

mod scheduler;

static DEPTH: AtomicU32 = AtomicU32::new(1);

/// Per-loop scheduler state owned by the caller.
///
/// C/C++ should allocate this struct (stack or heap) and keep it across calls.
/// It is valid to zero-initialize it (e.g. `KorithSchedulerState state = {0};`);
/// the scheduler will sanitize and clamp fields on every call.
///
/// C equivalent:
/// ```c
/// typedef struct KorithSchedulerState {
///   int speculative_depth;
///   int batch_size;
///   int baseline_ready;
/// } KorithSchedulerState;
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct KorithSchedulerState {
  /// Current speculative depth (caller-visible state).
  pub speculative_depth: c_int,
  /// Current batch size (caller-visible state).
  pub batch_size: c_int,
  /// Baseline warmup ready flag (0 = false, non-zero = true).
  pub baseline_ready: c_int,
}

/// Inputs (signals + hard limits) for a single scheduling decision.
///
/// C equivalent:
/// ```c
/// typedef struct KorithScheduleInput {
///   float   acceptance;
///   float   entropy;
///   float   current_tps;
///   float   engine_costs;
///   float   amf_reuse_score;
///   float   amf_avg_prefix_length;
///   float   amf_accept_rate;
///   float   amf_restore_cost_ms;
///   int     baseline_ready;
/// } KorithScheduleInput;
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct KorithScheduleInput {
  /// Speculative acceptance ratio or EMA, expected in [0, 1].
  pub acceptance: f32,
  /// Normalized entropy of the current token distribution, expected in [0, 1].
  pub entropy: f32,
  /// Current TPS delta (spec TPS vs baseline), expected to be finite.
  pub current_tps: f32,
  /// Aggregate engine cost estimate (normalized to seconds or arbitrary units).
  pub engine_costs: f32,
  /// AMF reuse score for the current prefix, expected in [0, 1].
  pub amf_reuse_score: f32,
  /// Average cached prefix length across AMF entries.
  pub amf_avg_prefix_length: f32,
  /// Historical AMF accept rate, expected in [0, 1].
  pub amf_accept_rate: f32,
  /// Average AMF restore cost in milliseconds.
  pub amf_restore_cost_ms: f32,
  /// Baseline warmup ready flag (0 = false, non-zero = true).
  pub baseline_ready: c_int,
}

/// Authoritative execution parameters for a single inference step.
///
/// C equivalent:
/// ```c
/// typedef struct KorithScheduleOutput {
///   int   desired_depth;
///   float stability_score;
///   int   throttle_flag;   // 0=default, 1=miss-budget, 2=hit-budget, 3=spec-hit-budget
/// } KorithScheduleOutput;
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct KorithScheduleOutput {
  /// Desired speculative depth (>= 1).
  pub desired_depth: c_int,
  /// Stability score in [0, 1].
  pub stability_score: f32,
  /// Budget profile: 0=default, 1=miss, 2=hit, 3=spec_hit.
  pub throttle_flag: c_int,
}

/// Computes authoritative execution parameters for the next inference step.
///
/// # Safety / ownership
/// - `state` must be either NULL or point to a valid, writable `KorithSchedulerState`.
/// - `input` must be either NULL or point to a valid, readable `KorithScheduleInput`.
/// - The scheduler never takes ownership of these pointers and never stores them.
/// - No heap memory is allocated or freed across this boundary.
///
/// # Threading
/// The function is deterministic. If the same `state` is shared across threads,
/// the caller must synchronize access (the scheduler mutates `*state`).
#[no_mangle]
pub extern "C" fn korith_schedule_step(
  state: *mut KorithSchedulerState,
  input: *const KorithScheduleInput,
) -> KorithScheduleOutput {
  let mut state_buf = KorithSchedulerState::default();
  let input_buf = KorithScheduleInput::default();

  let state_ref = if state.is_null() {
    &mut state_buf
  } else {
    // SAFETY: caller guarantees pointer validity.
    unsafe { &mut *state }
  };

  let input_ref = if input.is_null() {
    &input_buf
  } else {
    // SAFETY: caller guarantees pointer validity.
    unsafe { &*input }
  };

  scheduler::schedule_step(state_ref, input_ref)
}

#[no_mangle]
pub extern "C" fn korith_scheduler_update(
  entropy: f32,
  acceptance: f32,
  tps: f32,
) -> u32 {
  let mut state = KorithSchedulerState {
    speculative_depth: DEPTH.load(Ordering::Relaxed) as c_int,
    batch_size: 0,
    baseline_ready: 1,
  };
  let input = KorithScheduleInput {
    acceptance,
    entropy,
    current_tps: tps,
    engine_costs: 0.0,
    amf_reuse_score: 0.0,
    amf_avg_prefix_length: 0.0,
    amf_accept_rate: 0.0,
    amf_restore_cost_ms: 0.0,
    baseline_ready: 1,
  };

  let out = scheduler::schedule_step(&mut state, &input);
  let depth = out.desired_depth.max(1) as u32;
  DEPTH.store(depth, Ordering::Relaxed);
  depth
}

#[no_mangle]
pub extern "C" fn korith_scheduler_reset() {
  DEPTH.store(1, Ordering::Relaxed);
}
