fn main() {
    // Only run build script if openblas feature is enabled
    #[cfg(all(feature = "openblas", not(target_os = "macos")))]
    {
        // Try pkg-config first - check common pkg-config paths for OpenBLAS
        let target_arch = std::env::var("TARGET").unwrap_or_default();
        let is_aarch64 = target_arch.contains("aarch64");
        let is_x86_64 = target_arch.contains("x86_64");
        
        // Add common pkg-config paths for OpenBLAS
        let pkg_config_paths = if is_aarch64 {
            vec![
                "/usr/lib/aarch64-linux-gnu/openblas-pthread/pkgconfig",
                "/usr/lib/aarch64-linux-gnu/openblas-openmp/pkgconfig",
            ]
        } else if is_x86_64 {
            vec![
                "/usr/lib/x86_64-linux-gnu/openblas-pthread/pkgconfig",
                "/usr/lib/x86_64-linux-gnu/openblas-openmp/pkgconfig",
            ]
        } else {
            vec![]
        };
        
        // Try pkg-config - it should find OpenBLAS if PKG_CONFIG_PATH is set correctly
        // But we'll also check file system directly as fallback
        if pkg_config::Config::new().probe("openblas").is_ok() {
            println!("cargo:rustc-link-lib=openblas");
            return;
        }
        
        // If pkg-config didn't work, try finding the library file directly in common OpenBLAS locations
        for pkg_path in &pkg_config_paths {
            // Extract the lib directory from pkgconfig path
            if let Some(parent) = std::path::Path::new(pkg_path).parent() {
                let lib_path = parent.join("libopenblas.so");
                if lib_path.exists() {
                    if let Some(lib_parent) = lib_path.parent() {
                        println!("cargo:rustc-link-search=native={}", lib_parent.display());
                        println!("cargo:rustc-link-lib=dylib=openblas");
                        return;
                    }
                }
            }
        }

        // Fallback: try to find OpenBLAS via standard library paths
        // On Ubuntu/Debian, OpenBLAS may be in openblas-pthread subdirectory
        
        // Try common library search paths (prioritize arch-specific paths)
        let search_paths = if is_aarch64 {
            vec![
                "/usr/lib/aarch64-linux-gnu/openblas-pthread",
                "/usr/lib/aarch64-linux-gnu/openblas-openmp",
                "/usr/lib/aarch64-linux-gnu",
                "/usr/lib",
                "/usr/local/lib",
            ]
        } else if is_x86_64 {
            vec![
                "/usr/lib/x86_64-linux-gnu/openblas-pthread",
                "/usr/lib/x86_64-linux-gnu/openblas-openmp",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib",
                "/usr/local/lib",
            ]
        } else {
            vec![
                "/usr/lib",
                "/usr/local/lib",
            ]
        };

        // First, try to find libopenblas.so symlink directly
        let possible_lib_paths = vec![
            "/usr/lib/aarch64-linux-gnu/libopenblas.so",
            "/usr/lib/x86_64-linux-gnu/libopenblas.so",
            "/usr/lib/libopenblas.so",
        ];
        
        for lib_path_str in possible_lib_paths {
            let lib_path = std::path::Path::new(lib_path_str);
            if lib_path.exists() {
                // Found the library, add its parent directory to search path
                if let Some(parent) = lib_path.parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                    println!("cargo:rustc-link-lib=dylib=openblas");
                    println!("cargo:warning=Linked OpenBLAS from {}", lib_path_str);
                    return;
                }
            }
        }
        
        // Fallback: search in directories
        for path in search_paths {
            let lib_path = std::path::Path::new(path);
            if lib_path.exists() {
                // Check for OpenBLAS libraries - try different naming conventions
                if let Ok(entries) = std::fs::read_dir(lib_path) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if let Some(name_str) = name.to_str() {
                            // Check for various OpenBLAS naming patterns
                            if name_str.contains("openblas") {
                                if name_str.starts_with("libopenblas") && 
                                   (name_str.ends_with(".so") || name_str.ends_with(".a") || name_str.contains(".so.")) {
                                    // Extract library name (without lib prefix and extension)
                                    let lib_name = if name_str.starts_with("lib") {
                                        if let Some(dot_pos) = name_str.rfind('.') {
                                            &name_str[3..dot_pos]
                                        } else {
                                            &name_str[3..]
                                        }
                                    } else {
                                        continue;
                                    };
                                    
                                    // Add search path and link to the library
                                    println!("cargo:rustc-link-search=native={}", path);
                                    println!("cargo:rustc-link-lib=dylib={}", lib_name);
                                    println!("cargo:warning=Linked OpenBLAS from {}: {}", path, name_str);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Last resort: try linking with common OpenBLAS library names
        println!("cargo:warning=OpenBLAS not found via pkg-config or file search, trying common names");
        // Try linking to openblas (will fail if not found, but that's okay)
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}

