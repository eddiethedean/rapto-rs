# Linux ARM64 Benchmarking Alternatives to Docker

## Current Situation

Docker Desktop on macOS uses virtualization (QEMU) to run ARM64 Linux containers, which introduces significant performance overhead:
- **macOS native**: 5-8x faster than NumPy
- **Linux Docker**: 0.10-0.20x (50-200x slower than macOS)
- Even forced SIMD is 5-10x slower than NumPy in Docker

## Alternatives Ranked by Performance

### 1. Native Linux ARM64 Hardware ⭐⭐⭐⭐⭐ (Best Performance)

**Pros:**
- No virtualization overhead
- True performance measurements
- Full hardware feature access (perf counters, etc.)

**Cons:**
- Requires physical hardware
- Cost (if purchasing)
- Less convenient for local development

**Options:**
- **Raspberry Pi 5** (~$100): Good for development, but slower than Apple Silicon
- **AWS Graviton instances**: Native ARM64, pay-per-use
- **Cloud ARM64 VMs**: Various providers offer ARM64 instances
- **Colocation/borrowed hardware**: If available

**Recommendation**: Use for final validation and production benchmarks.

---

### 2. GitHub Actions Self-Hosted ARM64 Runners ⭐⭐⭐⭐ (Good for CI)

**Pros:**
- Native ARM64 performance
- Integrated into CI/CD
- Can use existing ARM64 hardware
- Free if you have hardware

**Cons:**
- Requires setting up and maintaining runner
- Need ARM64 hardware to host runner
- Security considerations for self-hosted runners

**Setup:**
```yaml
# .github/workflows/linux-arm64-bench.yml
jobs:
  benchmarks:
    runs-on: self-hosted
    runs-on: [self-hosted, linux, ARM64]
    steps:
      # ... same as current bench.yml
```

**Recommendation**: Best option for CI/CD if you have ARM64 hardware available.

---

### 3. AWS Graviton GitHub Actions ⭐⭐⭐⭐ (Good for CI, Costs Money)

**Pros:**
- Native ARM64 performance
- No hardware to maintain
- Reliable and consistent
- Pay-per-use

**Cons:**
- Costs money (~$0.0084/hour for t4g.small)
- Requires AWS account setup
- More complex setup

**Options:**
- Use `runs-on: ubuntu-latest` with matrix strategy for ARM64
- Or use AWS CodeBuild with Graviton instances
- Or use self-hosted runner on AWS Graviton EC2 instance

**Recommendation**: Good option if budget allows and you need reliable CI.

---

### 4. Linux VM (VMware/Parallels/VirtualBox) ⭐⭐⭐ (Better than Docker, but still virtualized)

**Pros:**
- Better performance than Docker Desktop
- Full Linux environment
- Better hardware access than containers
- Can use KVM acceleration on Linux host

**Cons:**
- Still has virtualization overhead
- Requires VM setup and maintenance
- More resource intensive
- Performance still won't match native

**Setup:**
- Install Ubuntu ARM64 in VM
- Share project directory via shared folder
- Run benchmarks natively in VM

**Recommendation**: Better than Docker for local development, but still not ideal for final benchmarks.

---

### 5. WSL2 (Windows Subsystem for Linux) ⭐⭐ (If on Windows)

**Pros:**
- Better integration than Docker
- Can use native Linux binaries
- Good for Windows development

**Cons:**
- Still virtualization overhead
- Windows-only
- Performance still limited

**Recommendation**: Only if you're on Windows and need Linux testing.

---

### 6. Podman/LXD/Other Container Runtimes ⭐⭐ (Similar to Docker)

**Pros:**
- Different container runtime
- Some have better performance characteristics
- Rootless by default (Podman)

**Cons:**
- Still containerization overhead
- Similar virtualization issues on macOS
- May not solve the fundamental problem

**Recommendation**: Unlikely to solve the performance issue since the problem is macOS virtualization, not Docker specifically.

---

## Recommended Approach

### Short-term (Development)
1. **Keep Docker for development** - It's convenient and works for testing correctness
2. **Document Docker limitations** - Make it clear that Docker benchmarks are not representative
3. **Focus on macOS benchmarks** - Use native macOS performance as the primary metric

### Medium-term (CI/CD)
1. **Set up GitHub Actions with ARM64 runner** (if hardware available)
   - Use self-hosted runner on ARM64 hardware
   - Or use AWS Graviton instances
2. **Add ARM64 benchmark job** to CI
   - Run on native ARM64 hardware
   - Compare against macOS results

### Long-term (Production)
1. **Validate on native Linux ARM64** before releases
2. **Maintain separate baselines**:
   - macOS (native)
   - Linux ARM64 (native)
   - Linux ARM64 Docker (for reference, but not target)

---

## Implementation Plan

### Option A: GitHub Actions Self-Hosted Runner (Recommended if hardware available)

1. Set up ARM64 hardware (Raspberry Pi 5, AWS Graviton, etc.)
2. Install GitHub Actions runner
3. Create new workflow file:
   ```yaml
   # .github/workflows/linux-arm64-bench.yml
   name: Linux ARM64 Benchmarks
   
   on:
     workflow_dispatch:
     schedule:
       - cron: "0 6 * * *"
   
   jobs:
     benchmarks:
       runs-on: self-hosted
       runs-on: [self-hosted, linux, ARM64]
       steps:
         - uses: actions/checkout@v4
         # ... rest of setup
   ```

### Option B: AWS Graviton (Recommended if no hardware)

1. Set up AWS account
2. Create GitHub Actions workflow using AWS CodeBuild or EC2
3. Run benchmarks on Graviton instances
4. Cost: ~$0.20 per benchmark run (estimated)

### Option C: Hybrid Approach

1. Use Docker for local development (convenience)
2. Use GitHub Actions ARM64 for CI (accuracy)
3. Document both in README with clear labels

---

## Cost Comparison

| Option | Setup Cost | Per-Run Cost | Performance |
|--------|-----------|--------------|-------------|
| Docker Desktop | Free | Free | Poor (virtualized) |
| Native Hardware | $100-500 | Free | Excellent |
| Self-Hosted Runner | $100-500 | Free | Excellent |
| AWS Graviton | Free | ~$0.20 | Excellent |
| Linux VM | Free | Free | Good (virtualized) |

---

## Conclusion

**For accurate Linux ARM64 benchmarks, you need native hardware.** Docker Desktop's virtualization overhead makes it unsuitable for performance testing, though it's fine for correctness testing.

**Recommended path forward:**
1. Continue using Docker for local development and correctness testing
2. Set up GitHub Actions with ARM64 runner (self-hosted or AWS Graviton) for CI benchmarks
3. Document that Docker benchmarks are not representative of native performance
4. Use native Linux ARM64 for final validation before releases

The code-level optimizations we've implemented are still valuable and will show their true performance on native hardware.

