//! Thermodynamic control primitives derived from holographic memory.
//!
//! This module is intentionally deterministic and allocation-free.

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct HoloEvent {
  pub step_id: u64,
  pub token_id: i32,
  pub thermo_depth_at_commit: u8,
  pub accepted: bool,
  pub acceptance_rate_at_commit: f32,
  pub entropy_at_commit: f32,
  pub speculative_cost: f32,
  pub tps_snapshot: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ThermoDecision {
  pub allow_speculation: bool,
  pub allowed_depth: u8,
  pub free_energy: f32,
}

/// Computes a simple "free energy" score from the most recent holographic event.
///
/// Equations:
/// - `F = acceptance_rate_at_commit - lambda * speculative_cost`
/// - `lambda = 0.5`
///
/// Decision rules:
/// - `allow_speculation = (F > 0.2)`
/// - `allowed_depth = 1` (for now)
pub fn evaluate_thermo(events: *const HoloEvent, count: usize) -> ThermoDecision {
  if events.is_null() || count == 0 {
    return ThermoDecision {
      allow_speculation: false,
      allowed_depth: 1,
      free_energy: 0.0,
    };
  }

  // SAFETY: caller promises `events` points to at least `count` readable entries.
  let slice = unsafe { std::slice::from_raw_parts(events, count) };

  // The holographic ring buffer may not be ordered; select the newest by `step_id`.
  let mut newest = &slice[0];
  for e in &slice[1..] {
    if e.step_id >= newest.step_id {
      newest = e;
    }
  }

  let mut acceptance = newest.acceptance_rate_at_commit;
  if !acceptance.is_finite() {
    acceptance = 0.0;
  } else {
    if acceptance < 0.0 {
      acceptance = 0.0;
    } else if acceptance > 1.0 {
      acceptance = 1.0;
    }
  }

  let mut cost = newest.speculative_cost;
  if !cost.is_finite() || cost < 0.0 {
    cost = 0.0;
  }

  let free_energy = acceptance - 0.5_f32 * cost;
  let allow = free_energy > 0.2_f32;

  ThermoDecision {
    allow_speculation: allow,
    allowed_depth: 1,
    free_energy,
  }
}
