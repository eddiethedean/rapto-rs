use libloading::Library;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

static PYTHON_INIT: Once = Once::new();

fn default_python() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // workspace root
    path.push(".venv");
    path.push("bin");
    path.push("python");
    path
}

fn resolve_python_executable() -> PathBuf {
    if let Ok(explicit) = std::env::var("RAPTORS_TEST_PYTHON") {
        let candidate = PathBuf::from(explicit);
        if candidate.is_file() {
            return candidate;
        }
    }
    let candidate = default_python();
    if candidate.is_file() {
        return candidate;
    }
    PathBuf::from("python3")
}

fn resolve_python_library(python: &Path) -> PathBuf {
    let script = r#"
import sysconfig, pathlib
libdir = sysconfig.get_config_var('LIBDIR')
name = sysconfig.get_config_var('INSTSONAME') or sysconfig.get_config_var('LDLIBRARY')
path = pathlib.Path(libdir) / name
print(path.resolve())
"#;
    let output = Command::new(python)
        .args(["-c", script])
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "Failed to execute {} to locate libpython: {err}",
                python.display()
            )
        });
    if !output.status.success() {
        panic!(
            "Python reported failure while locating libpython:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let path = String::from_utf8(output.stdout)
        .expect("python output should be valid utf-8")
        .trim()
        .to_owned();
    let path = PathBuf::from(path);
    if !path.exists() {
        panic!(
            "Resolved libpython path does not exist: {}\nSet RAPTORS_TEST_PYTHON to a python with development headers installed.",
            path.display()
        );
    }
    path
}

pub fn ensure_python_initialized() {
    PYTHON_INIT.call_once(|| {
        let python = resolve_python_executable();
        let libpython = resolve_python_library(&python);
        unsafe {
            Library::new(&libpython).unwrap_or_else(|err| {
                panic!(
                    "Failed to load libpython from {}: {err}",
                    libpython.display()
                )
            });
        }
        pyo3::prepare_freethreaded_python();
    });
}
