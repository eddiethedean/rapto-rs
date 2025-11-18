# GitHub Actions ARM64 Benchmarking

## Overview

We use GitHub Actions native ARM64 runners to benchmark Linux ARM64 performance without Docker virtualization overhead. This provides accurate performance measurements that reflect real-world Linux ARM64 performance.

## Workflow

The workflow file is located at `.github/workflows/linux-arm64-bench.yml` and runs:

1. **Baseline measurements** - Full benchmark suite with forced SIMD, auto dispatch, and disabled SIMD
2. **Comparison benchmarks** - Focused mean_axis0 benchmarks across multiple shapes and dtypes
3. **Performance reports** - Summary of results with comparison to expected Docker performance

## Triggering the Workflow

### Manual Trigger (Recommended for Testing)

1. Go to the **Actions** tab in GitHub
2. Select **Linux ARM64 Benchmarks** workflow
3. Click **Run workflow**
4. Select branch (usually `main`)
5. Click **Run workflow**

### Automatic Triggers

The workflow runs automatically on:
- **Weekly schedule**: Every Sunday at 6:00 AM UTC
- **Push to main**: When Rust or Python code changes
- **Pull requests**: When PRs modify relevant code paths

## Expected Results

### Performance Comparison

On native ARM64 (GitHub Actions), we expect:

| Metric | Docker (macOS) | GitHub Actions ARM64 | Improvement |
|--------|----------------|---------------------|-------------|
| **Forced SIMD** | 0.10-0.20x | **5-8x** (expected) | 25-80x faster |
| **Auto dispatch** | 0.10-0.20x | **3-6x** (expected) | 15-60x faster |
| **vs NumPy** | 5-10x slower | **5-8x faster** (expected) | 50-100x improvement |

### Why the Improvement?

- **No virtualization overhead**: Native ARM64 execution
- **Full hardware access**: All CPU features available
- **Better memory performance**: No container memory overhead
- **Native SIMD**: Direct hardware access without emulation

## Viewing Results

### In GitHub Actions

1. Go to **Actions** tab
2. Click on the workflow run
3. Expand **Generate performance report** step to see summary
4. Download **linux-arm64-benchmark-results** artifact for full JSON results

### Artifacts

The workflow uploads:
- `benchmarks/linux_investigation/baseline_*.json` - Full baseline measurements
- `benchmarks/results/github-actions-arm64/latest.json` - Comparison benchmarks

### PR Comments

If the workflow runs on a pull request, it will automatically post a comment with a summary of results.

## Comparing with Docker Results

To compare GitHub Actions results with Docker:

1. **Download artifacts** from both runs
2. **Load JSON files** in Python:
   ```python
   import json
   
   with open('docker_results.json') as f:
       docker = json.load(f)
   with open('github_actions_results.json') as f:
       github = json.load(f)
   
   # Compare specific operations
   for case in docker['cases']:
       shape = case['shape']
       dtype = case['dtype']
       # Find matching case in GitHub results
       # Compare speedup values
   ```

3. **Expected differences**:
   - GitHub Actions should be 25-80x faster for forced SIMD
   - GitHub Actions should be 15-60x faster for auto dispatch
   - GitHub Actions should match or exceed macOS performance

## Troubleshooting

### Workflow Fails to Start

- **Check runner availability**: ARM64 runners may have limited availability
- **Check repository settings**: Ensure workflows are enabled
- **Check branch**: Workflow may only run on `main` branch

### Build Failures

- **OpenBLAS not found**: Check PKG_CONFIG_PATH is set correctly
- **Rust build fails**: Check Rust toolchain version
- **Python import errors**: Check virtual environment setup

### Performance Not as Expected

- **Check runner type**: Ensure using `ubuntu-22.04-arm64` (not x86_64)
- **Check environment variables**: RAPTORS_BLAS, RAPTORS_SIMD should be set
- **Compare with Docker**: If still slow, may indicate code issues, not environment

## Cost

- **Public repositories**: Free (ARM64 runners are free for public repos)
- **Private repositories**: Included in GitHub plan (check your plan limits)

## Next Steps

1. **Run the workflow** manually to get baseline
2. **Compare results** with Docker benchmarks
3. **Update documentation** with actual performance numbers
4. **Set up alerts** if performance degrades

## Related Documentation

- [Linux Benchmarking Alternatives](linux_benchmarking_alternatives.md)
- [Linux Performance Optimization Summary](linux_optimization_final_summary.md)
- [Docker Setup](DOCKER_SETUP.md)

