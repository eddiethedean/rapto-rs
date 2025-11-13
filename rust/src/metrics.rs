use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug)]
pub struct BackendUsage {
    pub operation: &'static str,
    pub dtype: &'static str,
    pub backend: &'static str,
    pub count: u64,
}

type BackendKey = (&'static str, &'static str, &'static str);

fn backend_map() -> &'static Mutex<HashMap<BackendKey, u64>> {
    static COUNTS: OnceLock<Mutex<HashMap<BackendKey, u64>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn record_backend(operation: &'static str, dtype: &'static str, backend: &'static str) {
    if operation.is_empty() || dtype.is_empty() || backend.is_empty() {
        return;
    }
    if let Ok(mut guard) = backend_map().lock() {
        let entry = guard.entry((operation, dtype, backend)).or_insert(0);
        *entry = entry.saturating_add(1);
    }
}

pub fn snapshot() -> Vec<BackendUsage> {
    backend_map()
        .lock()
        .map(|guard| {
            guard
                .iter()
                .map(|(&(operation, dtype, backend), &count)| BackendUsage {
                    operation,
                    dtype,
                    backend,
                    count,
                })
                .collect()
        })
        .unwrap_or_default()
}
