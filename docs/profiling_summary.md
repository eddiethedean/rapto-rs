# Profiling Summary - NumPy vs Raptors Performance Analysis

## Completed Tasks

### Infrastructure Setup

1. ✅ **Docker Configuration**: Updated `docker-compose.bench.yml` with:
   - `privileged: true` for perf access
   - Volume mount: `./benchmarks/profiles:/workspace/profiles` for output persistence

2. ✅ **Profiling Directory**: Created `benchmarks/profiles/` directory for all profiling output

### Assembly Extraction

1. ✅ **NumPy Assembly**: Successfully extracted assembly from `_multiarray_umath.cpython-311-aarch64-linux-gnu.so`
2. ⚠️ **Raptors Assembly**: Raptors library location needs verification (using editable install)

### Findings

**Performance Gap**:
- Raptors: ~0.46ms (0.73x NumPy)
- NumPy: ~0.34ms
- Gap: ~30% slower

**Key Constraints**:
- Perf events not supported in Docker Desktop emulated environment
- Full cache profiling requires native Linux system
- Assembly extraction successful but library location varies with install method

## Next Steps

1. **Locate Raptors Compiled Library**: Find the actual .so file for Raptors (may be in target/ or site-packages)
2. **Extract NEON Instructions**: Once both libraries located, extract and compare NEON instruction patterns
3. **Analyze Assembly Patterns**: Compare load-store-add sequences between implementations
4. **Native Profiling**: Run perf stat on native Linux system for cache behavior analysis

## Files Generated

- `benchmarks/profiles/numpy_asm.txt` - NumPy assembly (971,486 lines extracted)
- `benchmarks/profiles/raptors_asm.txt` - Raptors assembly (228,594 lines extracted)
- `benchmarks/profiles/raptors_neon.txt` - Extracted NEON instructions from Raptors
- `benchmarks/profiles/numpy_neon.txt` - Extracted NEON instructions from NumPy

**Library Locations**:
- Raptors: `/workspace/src/python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so`
- NumPy: `/workspace/.venv/lib/python3.11/site-packages/numpy/_core/_multiarray_umath.cpython-311-aarch64-linux-gnu.so`

## Documentation

- `docs/numpy_profiling_results.md` - Detailed profiling results
- `docs/numpy_profiling_findings.md` - Assembly analysis findings

