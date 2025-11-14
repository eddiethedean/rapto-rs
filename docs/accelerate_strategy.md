# Matching Accelerate Framework Performance

## Current Situation

**NumPy's Advantage:**
- NumPy uses Apple's Accelerate framework (hand-tuned assembly for Apple Silicon)
- Achieves 0.288ms best-case for float32 @ 2048² scale
- Uses Accelerate through ufunc system (optimized C code → Accelerate)

**Raptors' Current State:**
- Uses Accelerate `vDSP_vsmul` function directly
- Performance: ~0.42-0.50ms (0.85-0.90x speedup)
- Gap: ~10-15% slower than NumPy's best-case

**Key Finding:**
- We already have Accelerate integration, but it doesn't match NumPy's performance
- This suggests either:
  1. Function call overhead
  2. NumPy uses Accelerate differently (in-place, different functions, or with optimizations)
  3. NumPy has additional optimizations beyond Accelerate

## Options to Match Accelerate

### Option 1: Optimize Accelerate Usage ⭐ RECOMMENDED

**What we have:**
- `vDSP_vsmul` (vector-scalar multiply) - already implemented
- `cblas_sscal` (BLAS scale) - available in `blas.rs`

**What we can try:**

1. **Use BLAS scale instead of vDSP:**
   - `cblas_sscal` might be better optimized than `vDSP_vsmul`
   - BLAS functions are often more aggressive in their optimizations
   - Test if `cblas_sscal` performs better than `vDSP_vsmul`

2. **Use in-place operations:**
   - NumPy might use in-place operations (`arr *= factor`)
   - Accelerate has in-place variants that might be faster
   - Test `cblas_sscal` with in-place (modify input array directly)

3. **Reduce function call overhead:**
   - Current: Rust → extern "C" → Accelerate
   - Potential overhead in FFI boundary
   - Try to minimize checks/bounds in wrapper

4. **Use Accelerate with specific flags/options:**
   - Accelerate might have optimization flags
   - Check if NumPy sets specific options

**Implementation:**
- Modify `accelerate_vsmul_f32` to try `cblas_sscal` first
- Add in-place version using `cblas_sscal` directly on the array
- Profile to see if BLAS is faster than vDSP

### Option 2: Use Different Accelerate Functions

**Available Accelerate functions:**
- `vDSP_vsmul` - Vector-scalar multiply (current)
- `vDSP_vmul` - Vector-vector multiply (for different patterns)
- `cblas_sscal` - BLAS scale (potentially better)
- `vDSP_vsadd` - Vector-scalar add (already used for addition)

**What to try:**
- Test `cblas_sscal` vs `vDSP_vsmul` for performance
- See if NumPy uses a different function internally
- Use `dtruss` to see which Accelerate function NumPy actually calls

### Option 3: Improve Our SIMD to Match Accelerate

**Current approach:**
- Hand-tuned NEON SIMD with 16× unrolling
- Aggressive prefetching
- Optimized instruction scheduling

**What more we can do:**
- Use inline assembly for complete control
- Reverse-engineer Accelerate's assembly (if possible)
- Use LLVM optimization passes more aggressively
- Try different unrolling factors

**Limitations:**
- Accelerate is proprietary, so we can't see its exact implementation
- We're already quite optimized (~88-90% of NumPy)
- Diminishing returns on further SIMD optimization

### Option 4: Use Apple's Other Performance Libraries

**Metal Performance Shaders (MPS):**
- GPU acceleration for large arrays
- Could potentially beat Accelerate for very large arrays
- Requires GPU programming knowledge
- Not suitable for small arrays (transfer overhead)

**Other Apple frameworks:**
- Core ML (machine learning, not relevant here)
- vImage (image processing, not array math)

**Verdict:** Not applicable for this use case

### Option 5: Cross-Platform Alternatives

**OpenBLAS:**
- Open-source BLAS library
- Available on all platforms
- On Apple Silicon, not as optimized as Accelerate
- Could be slower than Accelerate

**Intel MKL:**
- Intel Math Kernel Library
- Excellent on x86, but x86-only
- Not available for Apple Silicon

**BLIS:**
- Modern BLAS implementation
- Better than OpenBLAS but still not Accelerate-level on Apple Silicon

**Verdict:** None match Accelerate on Apple Silicon

### Option 6: Compiler-Level Optimizations

**What we've done:**
- `target-cpu=native` (enabled)
- LTO thin (enabled)
- `codegen-units=1` (enabled)
- `opt-level=3` (enabled)

**What more we can try:**
- Full LTO (might help, but increases build time significantly)
- Additional LLVM optimization passes
- Profile-guided optimization (PGO)
- Link-time optimization flags

## Recommended Strategy

### Immediate Actions (High Priority)

1. **Test BLAS scale (`cblas_sscal`) vs vDSP (`vDSP_vsmul`):**
   - Create comparison benchmark
   - See if BLAS performs better than vDSP
   - Implement if faster

2. **Try in-place BLAS scale:**
   - Modify array in-place using `cblas_sscal`
   - Avoids copy overhead
   - Might be what NumPy does

3. **Trace NumPy's actual Accelerate calls:**
   - Use `dtruss` to see which function NumPy calls
   - Verify if it uses `vDSP_vsmul` or `cblas_sscal`
   - Match NumPy's exact approach

### Medium Priority

4. **Profile-guided optimization (PGO):**
   - Compile with profiling
   - Run benchmarks
   - Recompile with profile data
   - Can improve performance by 10-20%

5. **Try full LTO:**
   - Change from thin LTO to full LTO
   - Increases build time but might improve performance
   - Test if worth the trade-off

### Long Term

6. **Continue SIMD optimization:**
   - Hand-tune assembly further
   - Try to get closer to Accelerate's performance
   - Accept that we may not match exactly

7. **Consider Metal Performance Shaders:**
   - For very large arrays (>4096²)
   - GPU acceleration could help
   - Requires significant implementation effort

## Implementation Priority

**Phase 1 (Quick Wins):**
1. Test `cblas_sscal` vs `vDSP_vsmul` (30 minutes)
2. Implement faster option (15 minutes)
3. Profile to verify improvement (15 minutes)

**Phase 2 (If needed):**
1. Trace NumPy's Accelerate calls (1 hour with sudo)
2. Match NumPy's exact approach (1 hour)

**Phase 3 (Longer term):**
1. Profile-guided optimization setup (2-3 hours)
2. Full LTO experimentation (1 hour)
3. Further SIMD tuning (ongoing)

## Conclusion

**Best approach:** Optimize our Accelerate usage first:
- Test BLAS vs vDSP
- Try in-place operations
- Match NumPy's exact Accelerate calls

**Why this works:**
- We already have Accelerate access
- NumPy uses Accelerate, so matching it is feasible
- Less effort than further SIMD optimization
- Likely to close the remaining gap

**Realistic expectation:**
- Match or exceed NumPy's best-case performance
- Or at least get within 1-2% (currently 10-15% gap)
- Accept that some variance is unavoidable

