# ✅ Rust Scientific Computing Roadmap  
*A Practical Plan to Reach a NumPy-Equivalent Ecosystem in Rust*

## Overview
The Rust scientific computing ecosystem is powerful but fragmented. Unlike Python’s NumPy + SciPy stack, Rust currently requires multiple crates to achieve similar functionality. This document outlines:

✅ Necessary crates  
✅ What each covers  
⚠️ Feature gaps vs. NumPy  
📌 Priority focus areas for future development

---

## 📦 Rust Crates Needed for NumPy Equivalence

| Capability (NumPy Feature) | Rust Crate(s) | Coverage | Gaps vs NumPy / SciPy |
|---|---:|:---:|---|
| **n-dimensional arrays & broadcasting** | `ndarray` | ✅ | Missing some advanced broadcasting convenience & ufunc richness |
| **Linear algebra (BLAS/LAPACK: solve, svd, eig)** | `ndarray-linalg`, `nalgebra`, `faer` | ✅ / ⚠️ | BLAS/LAPACK linking not seamless; advanced SciPy-level ops scattered |
| **`.npy` / `.npz` file support** | `ndarray-npy` | ✅ | Dtype coverage smaller |
| **FFT** | `rustfft` | ✅ | Not under a unified NumPy-like namespace |
| **Random number generation** | `rand`, `rand_distr` | ✅ | APIs less consolidated than `numpy.random.Generator` |
| **Advanced / fancy indexing** | `ndarray` | ⚠️ | Masking / fancy indexing less feature-rich |
| **Vectorized operations (ufunc equivalents)** | `ndarray` + misc crates | ⚠️ | NumPy’s large ufunc library not matched |
| **Einsum / tensor contraction** | community crates | ⚠️ | Not first-class; less ergonomic |
| **Autograd + GPU acceleration** | `tch`, `burn` | ⚠️ / ✅ | Split ecosystem; not NumPy-style (closer to PyTorch/JAX) |
| **Sparse arrays** | `sprs` | ⚠️ | Coverage less mature than SciPy sparse |
| **Complex dtypes** | `ndarray` + `num-complex` | ✅ / ⚠️ | Limited dtype casting / promotion semantics |
| **Memory mapping / zero-copy array sharing** | `memmap2` + `ndarray` | ⚠️ | Manual setup required |
| **High-level scientific routines (SciPy modules)** | Many small crates | ❌ | No unified solution for signal/stats/integration/ODEs/etc. |
| **Broad ecosystem maturity** | entire ecosystem | ❌ | Rust lacks a SciPy-scale community library set |

---

## 🔧 Minimal Starter Stack

Add these to `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15"
ndarray-linalg = { version = "0.12", features = ["openblas"] }
ndarray-npy = "0.7"
rand = "0.8"
rustfft = "6.0"

# Optional deep learning / GPU
tch = { version = "0.14", optional = true }
burn = { version = "0.18", optional = true }
```

This gives you:

| Category | Rust Solution |
|---|---|
| Core arrays ✅ | `ndarray` |
| Linear algebra ✅ | `ndarray-linalg` |
| I/O ✅ | `ndarray-npy` |
| Random ✅ | `rand` |
| FFT ✅ | `rustfft` |

---

## 🚀 Priority Work for Full NumPy Parity

| Priority | Goal | Why it matters |
|---|---|---|
| 🔥 #1 | Unified vectorized ufunc system | Bring ergonomic math + dtype promotion consistency |
| #2 | Expanded advanced indexing support | Essential for scientific workflows and ML preprocessing |
| #3 | Integrated SciPy-style high-level routines | Enables adoption in scientific/engineering domains |
| #4 | One-stop GPU + autograd foundation | Critical for modern numerical computing |
| #5 | Standardized dataset I/O & zero-copy sharing | Smooth interop with Python ecosystems |

---

## 📌 Project Vision Alignment

Rust can exceed NumPy—not just replicate it—by driving:
- Memory safety guaranteed at compile time
- Built-in SIMD optimizations
- Unified CPU + GPU tensor execution
- Zero-cost abstractions for HPC and ML

Your **RAPTO-RS** project (Rust Accelerated Parallel Tensor Operations) fits perfectly as a high-performance, Rust-native numerical array engine with:
- A modern lazy execution engine
- Auto-parallelization
- SIMD + GPU backends
- Unified ufunc semantics
- Python bridge for adoption if needed

---

## ✅ Summary

Rust already has **90% of NumPy’s raw power**, but it’s scattered.  
This roadmap + crate set gives a practical NumPy alternative today —  
and a foundation to **surpass** it tomorrow.
