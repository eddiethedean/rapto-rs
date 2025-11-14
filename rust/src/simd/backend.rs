use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct BackendInfo {
    pub lane_bits: usize,
    pub row_tile_f32: usize,
    pub col_tile_f32: usize,
    pub row_tile_f64: usize,
    pub col_tile_f64: usize,
    pub prefetched_rows: usize,
}

static BACKEND_INFO: OnceLock<BackendInfo> = OnceLock::new();

pub fn backend() -> &'static BackendInfo {
    BACKEND_INFO.get_or_init(detect_backend)
}

pub trait SimdBackend {
    fn info() -> &'static BackendInfo;

    fn lane_width_bits() -> usize {
        Self::info().lane_bits
    }

    fn lanes_f32() -> usize {
        (Self::lane_width_bits() / 32).max(1)
    }

    fn lanes_f64() -> usize {
        (Self::lane_width_bits() / 64).max(1)
    }

    fn row_tile_f32() -> usize {
        Self::info().row_tile_f32.max(32)
    }

    fn col_tile_f32() -> usize {
        Self::info().col_tile_f32.max(32)
    }

    fn row_tile_f64() -> usize {
        Self::info().row_tile_f64.max(16)
    }

    fn col_tile_f64() -> usize {
        Self::info().col_tile_f64.max(16)
    }

    fn prefetched_rows() -> usize {
        Self::info().prefetched_rows.max(1)
    }
}

pub struct ActiveBackend;

impl SimdBackend for ActiveBackend {
    fn info() -> &'static BackendInfo {
        backend()
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_backend() -> BackendInfo {
    use crate::simd::cpu;

    let caps = cpu::capabilities();
    let lanes = caps.lane_width_bits;
    BackendInfo {
        lane_bits: lanes,
        row_tile_f32: 128,
        col_tile_f32: lanes / 32 * 8,
        row_tile_f64: 64,
        col_tile_f64: lanes / 64 * 8,
        prefetched_rows: 4,
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_backend() -> BackendInfo {
    BackendInfo {
        lane_bits: 128,
        row_tile_f32: 128,
        col_tile_f32: 64,
        row_tile_f64: 64,
        col_tile_f64: 32,
        prefetched_rows: 4,
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_backend() -> BackendInfo {
    BackendInfo {
        lane_bits: 64,
        row_tile_f32: 64,
        col_tile_f32: 32,
        row_tile_f64: 32,
        col_tile_f64: 16,
        prefetched_rows: 2,
    }
}

pub fn row_tile_f32() -> usize {
    backend().row_tile_f32.max(32)
}

pub fn col_tile_f32() -> usize {
    backend().col_tile_f32.max(32)
}

pub fn row_tile_f64() -> usize {
    backend().row_tile_f64.max(16)
}

pub fn col_tile_f64() -> usize {
    backend().col_tile_f64.max(16)
}

pub fn lanes_f32() -> usize {
    (backend().lane_bits / 32).max(1)
}

pub fn lanes_f64() -> usize {
    (backend().lane_bits / 64).max(1)
}

pub fn prefetched_rows() -> usize {
    backend().prefetched_rows.max(1)
}

/// Prefetch a memory address for read access.
/// Uses architecture-specific prefetch instructions to bring data into cache.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn prefetch_read(addr: *const u8) {
    use std::arch::x86_64::_mm_prefetch;
    _mm_prefetch(addr as *const i8, _MM_HINT_T0);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn prefetch_read(addr: *const u8) {
    core::arch::asm!(
        "prfm pldl1keep, [{addr}]",
        addr = in(reg) addr,
        options(readonly, nostack)
    );
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub unsafe fn prefetch_read(_addr: *const u8) {
    // No-op for unsupported architectures
}

/// Prefetch a memory address for read access with a specific distance ahead.
#[inline]
pub unsafe fn prefetch_read_ahead<T>(base: *const T, distance: usize) {
    prefetch_read((base as *const u8).add(distance * std::mem::size_of::<T>()));
}
