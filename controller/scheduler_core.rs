//! Deterministic scheduler core (FFI-friendly).
//!
//! This module is intentionally small and policy-only. It takes precomputed
//! signals (entropy, EMA acceptance, and a proposed depth) and emits a final
//! depth plus an execution decision.
//!
//! FFI surface:
//! - `scheduler_step(input) -> output`
//! - All structs/enums are `#[repr(C)]` / `#[repr(i32)]` for stable layout.

use std::io::Write;
use std::sync::Mutex;

const HISTORY_LEN: usize = 8;

#[derive(Debug, Copy, Clone)]
struct PredictHistory {
  entropies: [f32; HISTORY_LEN],
  acceptances: [f32; HISTORY_LEN],
  len: usize,
  idx: usize,
}

impl PredictHistory {
  const fn new() -> Self {
    Self {
      entropies: [1.0; HISTORY_LEN],
      acceptances: [0.0; HISTORY_LEN],
      len: 0,
      idx: 0,
    }
  }

  fn push(&mut self, entropy: f32, acceptance: f32) {
    self.entropies[self.idx] = entropy;
    self.acceptances[self.idx] = acceptance;
    self.idx = (self.idx + 1) % HISTORY_LEN;
    if self.len < HISTORY_LEN {
      self.len += 1;
    }
  }

  fn all_stable(&self) -> bool {
    if self.len < HISTORY_LEN {
      return false;
    }
    for i in 0..HISTORY_LEN {
      if !(self.entropies[i] < 0.55_f32 && self.acceptances[i] > 0.98_f32) {
        return false;
      }
    }
    true
  }
}

static PRED_HISTORY: Mutex<PredictHistory> = Mutex::new(PredictHistory::new());

#[repr(i32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Decision {
  Commit = 0,
  Rollback = 1,
  Fallback = 2,
}

impl Decision {
  #[inline]
  fn as_str(self) -> &'static str {
    match self {
      Decision::Commit => "Commit",
      Decision::Rollback => "Rollback",
      Decision::Fallback => "Fallback",
    }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct SchedulerInput {
  pub entropy: f32,
  pub ema_acceptance: f32,
  pub proposed_depth: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SchedulerOutput {
  pub final_depth: i32,
  pub decision: Decision,
}

#[inline]
fn sanitize_entropy(v: f32) -> f32 {
  if !v.is_finite() {
    return 1.0;
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
fn sanitize_acceptance(v: f32) -> f32 {
  if !v.is_finite() {
    return 0.0;
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
fn clamp_depth(depth: i32) -> i32 {
  if depth < 1 {
    1
  } else {
    depth
  }
}

/// Pure scheduler policy (no IO, no side effects).
pub fn scheduler_step_pure(input: SchedulerInput) -> SchedulerOutput {
  let entropy = sanitize_entropy(input.entropy);
  let ema = sanitize_acceptance(input.ema_acceptance);
  let proposed = input.proposed_depth;

  if entropy < 0.6_f32 && ema > 0.97_f32 {
    let mut final_depth = proposed;
    if final_depth > 8 {
      final_depth = 8;
    }
    final_depth = clamp_depth(final_depth);
    return SchedulerOutput {
      final_depth,
      decision: Decision::Commit,
    };
  }

  if entropy < 0.8_f32 {
    return SchedulerOutput {
      final_depth: 1,
      decision: Decision::Fallback,
    };
  }

  SchedulerOutput {
    final_depth: 1,
    decision: Decision::Rollback,
  }
}

/// FFI entrypoint: deterministic scheduler step with per-call logging.
#[no_mangle]
pub extern "C" fn scheduler_step(input: SchedulerInput) -> SchedulerOutput {
  let entropy = sanitize_entropy(input.entropy);
  let ema = sanitize_acceptance(input.ema_acceptance);

  let mut proposed = clamp_depth(input.proposed_depth);
  if proposed > 8 {
    proposed = 8;
  }

  // Predictive depth ramping:
  // - Observe last HISTORY_LEN steps of (entropy, ema_acceptance).
  // - If the entire window is stable, nudge depth upward by +1 (max 8).
  // - If entropy spikes above 0.75, immediately reduce depth to 1.
  let stable_window = {
    let mut h = match PRED_HISTORY.lock() {
      Ok(g) => g,
      Err(poisoned) => poisoned.into_inner(),
    };
    h.push(entropy, ema);
    h.all_stable()
  };

  let mut ramp_up = false;
  if entropy > 0.75_f32 {
    proposed = 1;
  } else if stable_window {
    ramp_up = true;
    if proposed < 8 {
      proposed += 1;
    }
  }

  let out = scheduler_step_pure(SchedulerInput {
    entropy,
    ema_acceptance: ema,
    proposed_depth: proposed,
  });

  // Log every decision; ignore IO errors (e.g., broken pipe) so we never panic.
  let mut stderr = std::io::stderr();
  let _ = writeln!(
    &mut stderr,
    "[PREDICT] ramp_up={} depth={}",
    if ramp_up { "true" } else { "false" },
    out.final_depth
  );
  let _ = writeln!(
    &mut stderr,
    "[SCHEDULER] entropy={:.4} ema={:.4} proposed={} final={} decision={}",
    input.entropy,
    input.ema_acceptance,
    input.proposed_depth,
    out.final_depth,
    out.decision.as_str()
  );

  out
}
