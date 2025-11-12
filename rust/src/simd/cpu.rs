use std::fmt;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
pub struct SimdCapabilities {
    pub arch: &'static str,
    pub avx512: bool,
    pub avx2: bool,
    pub avx: bool,
    pub fma: bool,
    pub sse41: bool,
    pub neon: bool,
    pub sve: bool,
    pub lane_width_bits: usize,
}

impl SimdCapabilities {
    pub fn feature_level(&self) -> &'static str {
        if self.avx512 {
            "avx512"
        } else if self.avx2 {
            "avx2"
        } else if self.avx {
            "avx"
        } else if self.neon {
            if self.sve {
                "sve"
            } else {
                "neon"
            }
        } else if self.sse41 {
            "sse4.1"
        } else {
            "scalar"
        }
    }
}

impl fmt::Display for SimdCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{} (lane={}b, fma={}, sve={})",
            self.arch,
            self.feature_level(),
            self.lane_width_bits,
            self.fma,
            self.sve
        )
    }
}

static CAPABILITIES: OnceLock<SimdCapabilities> = OnceLock::new();

pub fn capabilities() -> &'static SimdCapabilities {
    CAPABILITIES.get_or_init(detect)
}

fn apply_env_overrides(mut caps: SimdCapabilities) -> SimdCapabilities {
    if let Ok(value) = std::env::var("RAPTORS_SIMD_MAX") {
        let value = value.trim().to_lowercase();
        match value.as_str() {
            "scalar" => {
                caps.avx512 = false;
                caps.avx2 = false;
                caps.avx = false;
                caps.neon = false;
                caps.sve = false;
                caps.sse41 = false;
                caps.fma = false;
            }
            "avx2" => {
                caps.avx512 = false;
            }
            "avx" => {
                caps.avx512 = false;
                caps.avx2 = false;
            }
            "sse4.1" | "sse41" => {
                caps.avx512 = false;
                caps.avx2 = false;
                caps.avx = false;
            }
            "neon" => {
                caps.avx512 = false;
                caps.avx2 = false;
                caps.avx = false;
                caps.sse41 = false;
            }
            _ => {}
        }
    }
    recompute_lane_width(&mut caps);
    caps
}

fn recompute_lane_width(caps: &mut SimdCapabilities) {
    caps.lane_width_bits = if caps.avx512 {
        512
    } else if caps.avx2 || caps.avx {
        256
    } else if caps.neon || caps.sse41 {
        128
    } else {
        64
    };
}

#[cfg(target_arch = "x86_64")]
fn detect() -> SimdCapabilities {
    let avx512 = std::arch::is_x86_feature_detected!("avx512f");
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    let avx = std::arch::is_x86_feature_detected!("avx");
    let fma = std::arch::is_x86_feature_detected!("fma");
    let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
    let lane_width_bits = if avx512 {
        512
    } else if avx2 {
        256
    } else if avx {
        256
    } else if sse41 {
        128
    } else {
        64
    };
    let caps = SimdCapabilities {
        arch: "x86_64",
        avx512,
        avx2,
        avx,
        fma,
        sse41,
        neon: false,
        sve: false,
        lane_width_bits,
    };
    apply_env_overrides(caps)
}

#[cfg(target_arch = "aarch64")]
fn detect() -> SimdCapabilities {
    let neon = true;
    let sve = std::arch::is_aarch64_feature_detected!("sve");
    let caps = SimdCapabilities {
        arch: "aarch64",
        avx512: false,
        avx2: false,
        avx: false,
        fma: true,
        sse41: false,
        neon,
        sve,
        lane_width_bits: if sve { 256 } else { 128 },
    };
    apply_env_overrides(caps)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> SimdCapabilities {
    let caps = SimdCapabilities {
        arch: "generic",
        avx512: false,
        avx2: false,
        avx: false,
        fma: false,
        sse41: false,
        neon: false,
        sve: false,
        lane_width_bits: 64,
    };
    apply_env_overrides(caps)
}
