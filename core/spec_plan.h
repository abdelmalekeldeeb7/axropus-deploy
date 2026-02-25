#pragma once

#include <algorithm>

namespace korith::core {

enum SpecDisableReason : int {
  SPEC_DISABLE_DEPTH_LT_2 = 0,
  SPEC_DISABLE_LANES_LT_2 = 1,
  SPEC_DISABLE_ENGINES_0 = 2,
  SPEC_DISABLE_CP_GATE_OFF = 3,
  SPEC_DISABLE_FAILSAFE = 4,
  SPEC_DISABLE_MEMORY_PRESSURE = 5,
  SPEC_DISABLE_CONTEXT_LIMIT = 6,
  SPEC_DISABLE_UNKNOWN = 7,
  SPEC_DISABLE_BASELINE_ONLY = 8,
};

struct SpecPlan {
  int depth = 1;
  int lanes = 1;
  float accept_min = 0.6f;
  bool enabled = false;
  bool allow_exec = false;
  bool allow_commit = false;
  bool hard_disable = false;
  int reason_code = SPEC_DISABLE_UNKNOWN;
};

struct SpecControlDecision {
  bool allow_exec = true;
  bool allow_commit = true;
  bool hard_disable = false;
};

struct SpecEnvOverrides {
  bool has_depth = false;
  int depth = 1;
  bool has_allow_exec = false;
  bool allow_exec = false;
  bool has_allow_commit = false;
  bool allow_commit = false;
};

inline SpecPlan resolve_spec_plan(const SpecPlan & sched,
                                  const SpecControlDecision & cp,
                                  const SpecEnvOverrides & env) {
  SpecPlan out = sched;

  out.allow_exec = out.enabled && out.depth >= 2;
  out.allow_commit = true;
  out.hard_disable = cp.hard_disable;

  if (!cp.allow_exec) {
    out.allow_exec = false;
  }
  if (!cp.allow_commit) {
    out.allow_commit = false;
  }

  if (env.has_depth) {
    out.depth = env.depth;
    out.lanes = std::max(1, out.depth);
    if (out.depth < 2) {
      out.enabled = false;
      out.reason_code = SPEC_DISABLE_DEPTH_LT_2;
    } else {
      out.enabled = true;
    }
  }
  if (env.has_allow_exec) {
    out.allow_exec = env.allow_exec;
  }
  if (env.has_allow_commit) {
    out.allow_commit = env.allow_commit;
  }

  if (out.depth < 2) {
    out.allow_exec = false;
  }
  if (out.lanes < 2) {
    out.allow_exec = false;
    if (out.reason_code == SPEC_DISABLE_UNKNOWN) {
      out.reason_code = SPEC_DISABLE_LANES_LT_2;
    }
  }

  if (out.hard_disable) {
    out.allow_exec = false;
    out.allow_commit = false;
    out.depth = 1;
    out.lanes = 1;
    out.enabled = false;
    out.reason_code = SPEC_DISABLE_FAILSAFE;
  }

  return out;
}

inline const char * spec_disable_reason_name(int code) {
  switch (code) {
    case SPEC_DISABLE_DEPTH_LT_2:
      return "DEPTH_LT_2";
    case SPEC_DISABLE_LANES_LT_2:
      return "LANES_LT_2";
    case SPEC_DISABLE_ENGINES_0:
      return "ENGINES_0";
    case SPEC_DISABLE_CP_GATE_OFF:
      return "CP_GATE_OFF";
    case SPEC_DISABLE_FAILSAFE:
      return "FAILSAFE";
    case SPEC_DISABLE_MEMORY_PRESSURE:
      return "MEMORY_PRESSURE";
    case SPEC_DISABLE_CONTEXT_LIMIT:
      return "CONTEXT_LIMIT";
    case SPEC_DISABLE_BASELINE_ONLY:
      return "BASELINE_ONLY";
    case SPEC_DISABLE_UNKNOWN:
    default:
      return "UNKNOWN";
  }
}

}  // namespace korith::core
