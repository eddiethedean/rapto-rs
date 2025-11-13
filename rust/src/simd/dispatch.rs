use std::collections::HashMap;
use std::env;
use std::sync::{Mutex, OnceLock};

use super::SimdCapabilities;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdMode {
    Auto,
    Force,
    Disable,
}

impl SimdMode {
    pub fn is_disabled(self) -> bool {
        matches!(self, SimdMode::Disable)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    Neon,
    Sve,
    Sse41,
    Avx,
    Avx2,
    Avx512,
}

impl SimdLevel {
    pub const fn label(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Neon => "neon",
            SimdLevel::Sve => "sve",
            SimdLevel::Sse41 => "sse4.1",
            SimdLevel::Avx => "avx",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
        }
    }

    pub fn supported(self, caps: &SimdCapabilities) -> bool {
        match self {
            SimdLevel::Scalar => true,
            SimdLevel::Neon => caps.neon,
            SimdLevel::Sve => caps.sve,
            SimdLevel::Sse41 => caps.sse41,
            SimdLevel::Avx => caps.avx,
            SimdLevel::Avx2 => caps.avx2,
            SimdLevel::Avx512 => caps.avx512,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Candidate<F> {
    pub level: SimdLevel,
    pub func: F,
}

impl<F> Candidate<F> {
    pub const fn new(level: SimdLevel, func: F) -> Self {
        Self { level, func }
    }
}

#[derive(Clone, Copy)]
pub struct DispatchResult<F> {
    pub level: SimdLevel,
    pub func: F,
}

pub struct DispatchTable<F: Copy + 'static> {
    name: &'static str,
    scalar: F,
    candidates: &'static [Candidate<F>],
}

impl<F: Copy + 'static> DispatchTable<F> {
    pub const fn new(name: &'static str, scalar: F, candidates: &'static [Candidate<F>]) -> Self {
        Self {
            name,
            scalar,
            candidates,
        }
    }

    pub fn resolve(&self, mode: SimdMode, caps: &SimdCapabilities) -> DispatchResult<F> {
        let result = if mode.is_disabled() {
            DispatchResult {
                level: SimdLevel::Scalar,
                func: self.scalar,
            }
        } else {
            let mut chosen: Option<DispatchResult<F>> = None;
            for candidate in self.candidates {
                if candidate.level.supported(caps) {
                    chosen = Some(DispatchResult {
                        level: candidate.level,
                        func: candidate.func,
                    });
                    break;
                }
            }
            chosen.unwrap_or(DispatchResult {
                level: SimdLevel::Scalar,
                func: self.scalar,
            })
        };
        record_selection(self.name, result.level);
        result
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

static MODE: OnceLock<SimdMode> = OnceLock::new();
static SELECTIONS: OnceLock<Mutex<HashMap<&'static str, SimdLevel>>> = OnceLock::new();

pub fn global_mode() -> SimdMode {
    *MODE.get_or_init(|| {
        if let Ok(value) = env::var("RAPTORS_SIMD") {
            match value.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "off" | "disable" => return SimdMode::Disable,
                "1" | "true" | "on" | "force" => return SimdMode::Force,
                _ => {}
            }
        }
        SimdMode::Auto
    })
}

fn record_selection(name: &'static str, level: SimdLevel) {
    if let Ok(mut guard) = SELECTIONS.get_or_init(|| Mutex::new(HashMap::new())).lock() {
        guard.insert(name, level);
    }
}

pub fn selection_snapshot() -> Vec<(String, SimdLevel)> {
    SELECTIONS
        .get()
        .and_then(|mutex| mutex.lock().ok().map(|map| map.clone()))
        .map(|map| {
            map.into_iter()
                .map(|(name, level)| (name.to_string(), level))
                .collect()
        })
        .unwrap_or_default()
}
