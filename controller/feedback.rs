//! Feedback signals and a stable (non-oscillatory) control law.
//!
//! Notes on stability:
//! - All noisy inputs are smoothed with EWMAs.
//! - Outputs are rate-limited and use deadbands/hysteresis.
//! - Speculative depth is updated at a slower cadence than the tick rate.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct FeedbackSample {
  pub tps_rolling: f64,
  pub power_w: Option<f64>,
  pub batch_latency: Duration,
  pub accept_ratio: Option<f64>,
  pub accept_ema: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum HintReason {
  None,
  PowerCap,
  Latency,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulingHints {
  pub suggested_sleep: Duration,
  pub reason: HintReason,
}

impl Default for SchedulingHints {
  fn default() -> Self {
    Self {
      suggested_sleep: Duration::from_millis(0),
      reason: HintReason::None,
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct ControlOutput {
  pub batch_tokens: i32,
  pub spec_depth: i32,
  pub hints: SchedulingHints,
}

#[derive(Debug, Clone)]
pub struct ControlConfig {
  /// Controller tick period (used only for defaults / scaling).
  pub tick: Duration,

  /// Target latency for one `engine_step(batch_tokens)` call.
  pub latency_target: Duration,
  /// Hysteresis band around the latency target.
  pub latency_deadband: Duration,

  /// Optional GPU power cap (watts). When exceeded, controller prioritizes reducing load.
  pub power_cap_w: Option<f64>,
  /// Power hysteresis band (watts).
  pub power_deadband_w: f64,

  /// Bounds for `engine_step(batch_tokens)`.
  pub batch_min: i32,
  pub batch_max: i32,

  /// Bounds for speculative depth (requires an engine setter to be effective).
  pub spec_min: i32,
  pub spec_max: i32,

  /// Update depth at a lower rate to avoid control chatter.
  pub spec_update_interval: Duration,

  /// Acceptance thresholds for depth control.
  pub accept_hi: f64,
  pub accept_lo: f64,
}

impl Default for ControlConfig {
  fn default() -> Self {
    Self {
      tick: Duration::from_millis(5),
      latency_target: Duration::from_millis(12),
      latency_deadband: Duration::from_millis(2),
      power_cap_w: None,
      power_deadband_w: 5.0,
      batch_min: 1,
      batch_max: 256,
      spec_min: 1,
      spec_max: 16,
      spec_update_interval: Duration::from_millis(75),
      accept_hi: 0.85,
      accept_lo: 0.60,
    }
  }
}

#[derive(Debug, Clone, Copy)]
struct Ewma {
  alpha: f64,
  value: Option<f64>,
}

impl Ewma {
  fn new(alpha: f64) -> Self {
    Self { alpha, value: None }
  }

  fn update(&mut self, x: f64) -> f64 {
    let v = match self.value {
      None => x,
      Some(prev) => self.alpha * x + (1.0 - self.alpha) * prev,
    };
    self.value = Some(v);
    v
  }

  fn get(&self) -> Option<f64> {
    self.value
  }
}

fn clamp_i32(x: i32, lo: i32, hi: i32) -> i32 {
  if x < lo {
    lo
  } else if x > hi {
    hi
  } else {
    x
  }
}

fn step_up(batch: i32) -> i32 {
  // Gentle ramp-up to avoid latency oscillation.
  std::cmp::max(1, batch / 16)
}

fn step_down(batch: i32) -> i32 {
  // Faster backoff than ramp-up.
  std::cmp::max(1, batch / 8)
}

#[derive(Debug)]
pub struct Controller {
  cfg: ControlConfig,

  // Smoothed signals.
  tps_ewma: Ewma,
  latency_ms_ewma: Ewma,
  power_w_ewma: Ewma,
  accept_ewma: Ewma,

  // Current outputs.
  batch_tokens: i32,
  spec_depth: i32,

  last_spec_update: Instant,
}

impl Controller {
  pub fn new(cfg: ControlConfig, initial_batch_tokens: i32, initial_spec_depth: i32) -> Self {
    let now = Instant::now();
    Self {
      tps_ewma: Ewma::new(0.20),
      latency_ms_ewma: Ewma::new(0.25),
      power_w_ewma: Ewma::new(0.15),
      accept_ewma: Ewma::new(0.10),
      batch_tokens: clamp_i32(initial_batch_tokens, cfg.batch_min, cfg.batch_max),
      spec_depth: clamp_i32(initial_spec_depth, cfg.spec_min, cfg.spec_max),
      cfg,
      last_spec_update: now,
    }
  }

  pub fn current(&self) -> ControlOutput {
    ControlOutput {
      batch_tokens: self.batch_tokens,
      spec_depth: self.spec_depth,
      hints: SchedulingHints::default(),
    }
  }

  pub fn update(&mut self, fb: FeedbackSample) -> ControlOutput {
    let latency_ms = fb.batch_latency.as_secs_f64() * 1e3;
    let tps = if fb.tps_rolling.is_finite() && fb.tps_rolling >= 0.0 {
      fb.tps_rolling
    } else {
      0.0
    };

    let latency_ms_s = self.latency_ms_ewma.update(latency_ms);
    let _tps_s = self.tps_ewma.update(tps);
    if let Some(p) = fb.power_w {
      if p.is_finite() && p >= 0.0 {
        let _ = self.power_w_ewma.update(p);
      }
    }
    if let Some(a) = fb.accept_ema.or(fb.accept_ratio) {
      if a.is_finite() && a >= 0.0 && a <= 1.0 {
        let _ = self.accept_ewma.update(a);
      }
    }

    let target_ms = self.cfg.latency_target.as_secs_f64() * 1e3;
    let deadband_ms = self.cfg.latency_deadband.as_secs_f64() * 1e3;
    let latency_hi = target_ms + deadband_ms;
    let latency_lo = (target_ms - deadband_ms).max(0.0);

    let power_over = match (self.cfg.power_cap_w, self.power_w_ewma.get()) {
      (Some(cap), Some(p)) if cap > 0.0 => p > cap + self.cfg.power_deadband_w,
      _ => false,
    };

    let latency_over = latency_ms_s > latency_hi;
    let latency_under = latency_ms_s < latency_lo;

    // --- Batch size control (fast loop) ---
    if power_over || latency_over {
      self.batch_tokens = clamp_i32(self.batch_tokens - step_down(self.batch_tokens), self.cfg.batch_min, self.cfg.batch_max);
    } else if latency_under {
      self.batch_tokens = clamp_i32(self.batch_tokens + step_up(self.batch_tokens), self.cfg.batch_min, self.cfg.batch_max);
    }

    // --- Spec depth control (slow loop) ---
    let now = Instant::now();
    if now.duration_since(self.last_spec_update) >= self.cfg.spec_update_interval {
      if let Some(a) = self.accept_ewma.get() {
        // Only increase depth when we're not constrained; otherwise prioritize stability.
        if !power_over && !latency_over && a >= self.cfg.accept_hi {
          self.spec_depth = clamp_i32(self.spec_depth + 1, self.cfg.spec_min, self.cfg.spec_max);
          self.last_spec_update = now;
        } else if power_over || latency_over || a <= self.cfg.accept_lo {
          // Back off more aggressively if acceptance collapses.
          let delta = if a <= 0.35 { 2 } else { 1 };
          self.spec_depth = clamp_i32(self.spec_depth - delta, self.cfg.spec_min, self.cfg.spec_max);
          self.last_spec_update = now;
        }
      }
    }

    // --- Scheduling hints (for multi-stream schedulers) ---
    let mut hints = SchedulingHints::default();
    if power_over {
      hints.reason = HintReason::PowerCap;
      hints.suggested_sleep = Duration::from_millis(1);
    } else if latency_over {
      hints.reason = HintReason::Latency;
      hints.suggested_sleep = Duration::from_millis(0);
    }

    ControlOutput {
      batch_tokens: self.batch_tokens,
      spec_depth: self.spec_depth,
      hints,
    }
  }
}

// ------------------------------ Power sensing -----------------------------

pub trait PowerSensor {
  fn sample_watts(&mut self) -> Option<f64>;
}

#[derive(Debug, Default)]
pub struct NullPowerSensor;

impl PowerSensor for NullPowerSensor {
  fn sample_watts(&mut self) -> Option<f64> {
    None
  }
}

#[derive(Debug, Clone)]
pub struct SysfsPowerSensor {
  path: PathBuf,
  scale_to_watts: f64,
}

impl SysfsPowerSensor {
  /// Attempts to discover a GPU power file in sysfs (Linux hwmon).
  ///
  /// Common units:
  /// - `power*_average` / `power*_input` are typically in microwatts.
  pub fn discover() -> Option<Self> {
    let hwmon = Path::new("/sys/class/hwmon");
    let dirs = fs::read_dir(hwmon).ok()?;
    let mut fallback: Option<Self> = None;
    for ent in dirs.flatten() {
      let dir = ent.path();
      let name = fs::read_to_string(dir.join("name")).unwrap_or_default().to_lowercase();

      // Prefer NVIDIA, but accept any if nothing else exists.
      let is_preferred = name.contains("nvidia");

      for file in ["power1_average", "power1_input"] {
        let p = dir.join(file);
        if p.exists() {
          let sensor = Self {
            path: p,
            scale_to_watts: 1e-6, // uW -> W
          };
          if is_preferred {
            return Some(sensor);
          }
          if fallback.is_none() {
            fallback = Some(sensor);
          }
          break;
        }
      }
    }
    fallback
  }

  fn read_raw(&self) -> Option<i64> {
    let s = fs::read_to_string(&self.path).ok()?;
    s.trim().parse::<i64>().ok()
  }
}

impl PowerSensor for SysfsPowerSensor {
  fn sample_watts(&mut self) -> Option<f64> {
    let raw = self.read_raw()?;
    if raw <= 0 {
      return None;
    }
    Some((raw as f64) * self.scale_to_watts)
  }
}

/// Tries to create the best available power sensor:
/// - NVML (fast, accurate) if available
/// - hwmon sysfs fallback
/// - otherwise a `NullPowerSensor`
pub fn default_power_sensor(gpu_index: u32) -> Box<dyn PowerSensor + Send> {
  #[cfg(target_os = "linux")]
  if let Some(nvml) = NvmlPowerSensor::new(gpu_index) {
    return Box::new(nvml);
  }

  if let Some(sysfs) = SysfsPowerSensor::discover() {
    return Box::new(sysfs);
  }

  Box::new(NullPowerSensor)
}

// ------------------------------ NVML (optional) ---------------------------

#[cfg(target_os = "linux")]
mod nvml {
  use std::ffi::{CStr, CString};
  use std::os::raw::{c_char, c_int, c_void};
  use std::ptr::NonNull;

  // POSIX dlfcn (used to avoid static linking to NVML).
  #[link(name = "dl")]
  extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
  }

  const RTLD_NOW: c_int = 2;
  const NVML_SUCCESS: i32 = 0;

  #[repr(C)]
  pub struct nvmlDevice_st {
    _private: [u8; 0],
  }
  pub type nvmlDevice_t = *mut nvmlDevice_st;

  pub type NvmlInitV2 = unsafe extern "C" fn() -> i32;
  pub type NvmlShutdown = unsafe extern "C" fn() -> i32;
  pub type NvmlDeviceGetHandleByIndexV2 = unsafe extern "C" fn(index: u32, device: *mut nvmlDevice_t) -> i32;
  pub type NvmlDeviceGetPowerUsage = unsafe extern "C" fn(device: nvmlDevice_t, power_mw: *mut u32) -> i32;
  pub type NvmlErrorString = unsafe extern "C" fn(result: i32) -> *const c_char;

  pub struct NvmlLib {
    pub handle: NonNull<c_void>,
    pub init_v2: NvmlInitV2,
    pub shutdown: NvmlShutdown,
    pub device_get_handle_by_index_v2: NvmlDeviceGetHandleByIndexV2,
    pub device_get_power_usage: NvmlDeviceGetPowerUsage,
    pub error_string: Option<NvmlErrorString>,
  }

  impl NvmlLib {
    pub fn open() -> Option<Self> {
      unsafe {
        let open_by_name = |soname: &'static str| -> Option<NonNull<c_void>> {
          let name = CString::new(soname).ok()?;
          let _ = dlerror();
          let h = dlopen(name.as_ptr(), RTLD_NOW);
          NonNull::new(h)
        };

        let handle = open_by_name("libnvidia-ml.so.1").or_else(|| open_by_name("libnvidia-ml.so"))?;

        let sym = |s: &'static str| -> Option<*mut c_void> {
          let cs = CString::new(s).ok()?;
          let _ = dlerror();
          let p = dlsym(handle.as_ptr(), cs.as_ptr());
          let err = dlerror();
          if !err.is_null() || p.is_null() {
            return None;
          }
          Some(p)
        };

        let init_v2_p = match sym("nvmlInit_v2") {
          Some(p) => p,
          None => {
            let _ = dlclose(handle.as_ptr());
            return None;
          }
        };
        let shutdown_p = match sym("nvmlShutdown") {
          Some(p) => p,
          None => {
            let _ = dlclose(handle.as_ptr());
            return None;
          }
        };
        let get_handle_p = match sym("nvmlDeviceGetHandleByIndex_v2") {
          Some(p) => p,
          None => {
            let _ = dlclose(handle.as_ptr());
            return None;
          }
        };
        let power_usage_p = match sym("nvmlDeviceGetPowerUsage") {
          Some(p) => p,
          None => {
            let _ = dlclose(handle.as_ptr());
            return None;
          }
        };

        let init_v2: NvmlInitV2 = std::mem::transmute_copy(&init_v2_p);
        let shutdown: NvmlShutdown = std::mem::transmute_copy(&shutdown_p);
        let device_get_handle_by_index_v2: NvmlDeviceGetHandleByIndexV2 = std::mem::transmute_copy(&get_handle_p);
        let device_get_power_usage: NvmlDeviceGetPowerUsage = std::mem::transmute_copy(&power_usage_p);
        let error_string: Option<NvmlErrorString> = sym("nvmlErrorString").map(|p| std::mem::transmute_copy(&p));

        Some(Self {
          handle,
          init_v2,
          shutdown,
          device_get_handle_by_index_v2,
          device_get_power_usage,
          error_string,
        })
      }
    }

    pub fn fmt_err(&self, rc: i32) -> String {
      if let Some(f) = self.error_string {
        unsafe {
          let p = f(rc);
          if !p.is_null() {
            return CStr::from_ptr(p).to_string_lossy().into_owned();
          }
        }
      }
      format!("nvml error {rc}")
    }
  }

  impl Drop for NvmlLib {
    fn drop(&mut self) {
      unsafe {
        let _ = dlclose(self.handle.as_ptr());
      }
    }
  }

  pub struct NvmlCtx {
    pub lib: NvmlLib,
    pub dev: nvmlDevice_t,
  }

  impl NvmlCtx {
    pub fn init(gpu_index: u32) -> Option<Self> {
      let lib = NvmlLib::open()?;
      let rc = unsafe { (lib.init_v2)() };
      if rc != NVML_SUCCESS {
        return None;
      }

      let mut dev: nvmlDevice_t = std::ptr::null_mut();
      let rc = unsafe { (lib.device_get_handle_by_index_v2)(gpu_index, &mut dev as *mut nvmlDevice_t) };
      if rc != NVML_SUCCESS || dev.is_null() {
        unsafe { (lib.shutdown)() };
        return None;
      }

      Some(Self { lib, dev })
    }

    pub fn power_watts(&self) -> Option<f64> {
      let mut mw: u32 = 0;
      let rc = unsafe { (self.lib.device_get_power_usage)(self.dev, &mut mw as *mut u32) };
      if rc != NVML_SUCCESS {
        return None;
      }
      Some((mw as f64) / 1000.0)
    }
  }

  impl Drop for NvmlCtx {
    fn drop(&mut self) {
      unsafe {
        let _ = (self.lib.shutdown)();
      }
    }
  }
}

#[cfg(target_os = "linux")]
pub struct NvmlPowerSensor {
  ctx: nvml::NvmlCtx,
}

#[cfg(target_os = "linux")]
impl NvmlPowerSensor {
  pub fn new(gpu_index: u32) -> Option<Self> {
    let ctx = nvml::NvmlCtx::init(gpu_index)?;
    Some(Self { ctx })
  }
}

#[cfg(target_os = "linux")]
impl PowerSensor for NvmlPowerSensor {
  fn sample_watts(&mut self) -> Option<f64> {
    self.ctx.power_watts()
  }
}
