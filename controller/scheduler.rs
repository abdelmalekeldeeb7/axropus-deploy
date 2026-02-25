//! Deterministic scheduler policy (no IO, no allocation).
//!
//! This module is pure logic. All state is stored in `KorithSchedulerState`,
//! which is owned and managed by the C/C++ caller.

use crate::{KorithScheduleInput, KorithScheduleOutput, KorithSchedulerState};
use libc::c_char;
use std::os::raw::c_int;

extern "C" {
  static mut stderr: *mut libc::FILE;
}

const PROBE_INTERVAL: u32 = 64;
const HYSTERESIS_STEPS: u32 = 32;
const STARVATION_STEPS: u32 = 96;

#[inline]
fn clamp_c_int(v: c_int, lo: c_int, hi: c_int) -> c_int {
  if hi < lo {
    return lo;
  }
  if v < lo {
    lo
  } else if v > hi {
    hi
  } else {
    v
  }
}

#[inline]
fn sanitize_unit_f32(v: f32, default: f32) -> f32 {
  if !v.is_finite() {
    return default;
  }
  if v < 0.0 {
    0.0
  } else if v > 1.0 {
    1.0
  } else {
    v
  }
}

#[inline]
fn sanitize_nonneg_f32(v: f32, default: f32) -> f32 {
  if !v.is_finite() {
    return default;
  }
  if v < 0.0 {
    0.0
  } else {
    v
  }
}

#[inline]
fn clamp_unit_f32(v: f32) -> f32 {
  if v < 0.0 {
    0.0
  } else if v > 1.0 {
    1.0
  } else {
    v
  }
}

#[inline]
fn sanitize_f32(v: f32, default: f32) -> f32 {
  if v.is_finite() {
    v
  } else {
    default
  }
}

#[inline]
fn unpack_counters(v: c_int) -> (u32, u32, u32) {
  let raw = v as u32;
  let step = raw & 0xFFFF;
  let last_change = (raw >> 16) & 0xFF;
  let miss_streak = (raw >> 24) & 0xFF;
  (step, last_change, miss_streak)
}

#[inline]
fn pack_counters(step: u32, last_change: u32, miss_streak: u32) -> c_int {
  let raw = ((miss_streak & 0xFF) << 24) | ((last_change & 0xFF) << 16) | (step & 0xFFFF);
  raw as c_int
}

#[inline]
unsafe fn log_decision(
  depth: c_int,
  stability: f32,
  throttle: c_int,
  lane: *const c_char,
  reason: *const c_char,
) {
  let fmt = b"[SCHEDULER_DECISION] depth=%d stability=%.3f throttle=%d lane=%s reason=%s\n\0";
  let _ = libc::fprintf(
    stderr,
    fmt.as_ptr() as *const c_char,
    depth,
    stability as f64,
    throttle,
    lane,
    reason,
  );
}

#[inline]
unsafe fn log_probe() {
  let fmt = b"[SCHED_PROBE] depth=2\n\0";
  let _ = libc::fprintf(stderr, fmt.as_ptr() as *const c_char);
}

#[inline]
unsafe fn log_hysteresis_blocked() {
  let fmt = b"[SCHED] hysteresis_blocked\n\0";
  let _ = libc::fprintf(stderr, fmt.as_ptr() as *const c_char);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LaneClass {
  SpecHit,
  Hit,
  Miss,
}

#[inline]
fn classify_lane(acceptance: f32, entropy: f32, tps_delta: f32, reuse_score: f32) -> LaneClass {
  let replay_local = reuse_score >= 0.70;
  let stable_decode = acceptance >= 0.78 && entropy <= 0.58 && tps_delta >= -0.05;
  if replay_local && stable_decode {
    LaneClass::SpecHit
  } else if replay_local || (acceptance >= 0.68 && entropy <= 0.68) {
    LaneClass::Hit
  } else {
    LaneClass::Miss
  }
}

#[inline]
fn lane_name_ptr(lane: LaneClass) -> *const c_char {
  match lane {
    LaneClass::SpecHit => b"SPEC_HIT\0".as_ptr() as *const c_char,
    LaneClass::Hit => b"HIT\0".as_ptr() as *const c_char,
    LaneClass::Miss => b"MISS\0".as_ptr() as *const c_char,
  }
}

pub(crate) fn schedule_step(
  state: &mut KorithSchedulerState,
  input: &KorithScheduleInput,
) -> KorithScheduleOutput {
  state.baseline_ready = if input.baseline_ready != 0 { 1 } else { 0 };
  if state.baseline_ready == 0 {
    state.speculative_depth = 1;
    return KorithScheduleOutput {
      desired_depth: 1,
      stability_score: 0.0,
      throttle_flag: 1, // Warmup defaults to MISS budget until baseline is stable.
    };
  }

  // Sanitize signals.
  let acceptance = sanitize_unit_f32(input.acceptance, 0.0);
  let entropy = sanitize_unit_f32(input.entropy, 1.0);
  let tps_delta = sanitize_f32(input.current_tps, 0.0);
  let _engine_costs = sanitize_nonneg_f32(input.engine_costs, 0.0);
  let reuse_score = sanitize_unit_f32(input.amf_reuse_score, 0.0);
  let _avg_prefix_len = sanitize_nonneg_f32(input.amf_avg_prefix_length, 0.0);
  let _amf_accept = sanitize_unit_f32(input.amf_accept_rate, 0.0);
  let _restore_cost_ms = sanitize_nonneg_f32(input.amf_restore_cost_ms, 0.0);

  // Load + sanitize state (caller may zero-init).
  let max_depth = 16;
  let mut depth = clamp_c_int(state.speculative_depth, 1, max_depth);
  let (mut step, mut last_change, mut miss_streak) = unpack_counters(state.batch_size);
  step = step.wrapping_add(1) & 0xFFFF;

  let lane = classify_lane(acceptance, entropy, tps_delta, reuse_score);
  if lane == LaneClass::Miss {
    miss_streak = (miss_streak + 1).min(255);
  } else {
    miss_streak = 0;
  }

  let mut reason: *const c_char = b"hold\0".as_ptr() as *const c_char;
  let mut force_down = false;
  let lane_cap: c_int;
  let throttle_flag: c_int;

  match lane {
    LaneClass::SpecHit => {
      lane_cap = 16;
      throttle_flag = 3; // Aggressive decode budget.
      if acceptance >= 0.90 && entropy <= 0.45 && tps_delta >= -0.02 {
        depth = depth.saturating_add(2);
        reason = b"spec_hit_up2\0".as_ptr() as *const c_char;
      } else if acceptance >= 0.80 && entropy <= 0.58 && tps_delta >= -0.05 {
        depth = depth.saturating_add(1);
        reason = b"spec_hit_up1\0".as_ptr() as *const c_char;
      } else if acceptance < 0.62 || entropy > 0.82 || tps_delta < -0.15 {
        depth = depth.saturating_sub(2).max(2);
        force_down = true;
        reason = b"spec_hit_down\0".as_ptr() as *const c_char;
      }
    }
    LaneClass::Hit => {
      lane_cap = 12;
      throttle_flag = 2; // Balanced decode budget.
      if acceptance >= 0.82 && entropy <= 0.62 && tps_delta >= -0.03 {
        depth = depth.saturating_add(1);
        reason = b"hit_up\0".as_ptr() as *const c_char;
      } else if acceptance < 0.58 || entropy > 0.84 || tps_delta < -0.14 {
        depth = depth.saturating_sub(1).max(2);
        force_down = true;
        reason = b"hit_down\0".as_ptr() as *const c_char;
      }
    }
    LaneClass::Miss => {
      lane_cap = 3;
      throttle_flag = 1; // Conservative decode budget for bounded miss lane.
      if acceptance < 0.50 || entropy > 0.85 || tps_delta < -0.10 {
        depth = 2;
        reason = b"miss_cap\0".as_ptr() as *const c_char;
      } else {
        depth = depth.min(3);
        reason = b"miss_hold\0".as_ptr() as *const c_char;
      }

      // Starvation guard: periodically allow a small probe expansion in MISS lane
      // when quality signals recover, so the lane can re-enter a productive regime.
      if miss_streak >= STARVATION_STEPS && acceptance >= 0.60 && entropy <= 0.70 {
        depth = 3;
        reason = b"miss_probe\0".as_ptr() as *const c_char;
        miss_streak = 0;
      }
      force_down = true;
    }
  }

  depth = clamp_c_int(depth, 1, lane_cap);

  // Periodic cold-start probe to prevent permanent depth collapse.
  let probe_ready = depth == 1 && (step % PROBE_INTERVAL == 0) && tps_delta >= 0.0;
  let mut probed = false;
  if probe_ready && depth == 1 {
    depth = 2;
    reason = b"probe\0".as_ptr() as *const c_char;
    probed = true;
    force_down = false;
  }

  depth = clamp_c_int(depth, 1, max_depth);

  if depth != state.speculative_depth {
    let step8 = step & 0xFF;
    let delta = step8.wrapping_sub(last_change) & 0xFF;
    if !probed && !force_down && delta < HYSTERESIS_STEPS {
      depth = state.speculative_depth;
      reason = b"hold\0".as_ptr() as *const c_char;
      unsafe { log_hysteresis_blocked() };
    } else {
      last_change = step8;
    }
  }

  let stability = clamp_unit_f32(acceptance * (1.0 - entropy));

  if probed && depth == 2 {
    unsafe { log_probe() };
  }

  // SAFETY: fixed NUL-terminated strings passed to libc.
  unsafe { log_decision(depth, stability, throttle_flag, lane_name_ptr(lane), reason) };

  // Persist caller-owned state for the next call.
  state.speculative_depth = depth;
  state.batch_size = pack_counters(step, last_change, miss_streak);

  KorithScheduleOutput {
    desired_depth: depth,
    stability_score: stability,
    throttle_flag,
  }
}
