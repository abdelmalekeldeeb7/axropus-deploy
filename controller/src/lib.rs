use std::sync::atomic::{AtomicU32, Ordering};

static DEPTH: AtomicU32 = AtomicU32::new(2);

#[no_mangle]
pub extern "C" fn korith_scheduler_update(
    _entropy: f32,
    _acceptance: f32,
    _tps: f32,
) -> u32 {
    DEPTH.store(2, Ordering::Relaxed);
    2
}

#[no_mangle]
pub extern "C" fn korith_scheduler_reset() {
    DEPTH.store(2, Ordering::Relaxed);
}
