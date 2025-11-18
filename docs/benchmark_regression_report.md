# Benchmark Regression Report (2025-11-17)

Comparisons use `benchmarks/docker_results/final_optimized_20251116-204150` as the baseline.

## Speedup Deltas

| Shape | Dtype | Op | Baseline Speedup | 20251117-101729 | Δ vs Base | 20251117-102125 | Δ vs Base |
|-------|-------|----|------------------|-----------------|-----------|-----------------|-----------|
| (512, 512) | float32 | broadcast_add | 1.38 | 0.79 | −0.59 | 0.63 | −0.74 |
| (512, 512) | float32 | mean | 1.25 | 3.60 | +2.35 | 2.97 | +1.72 |
| (512, 512) | float32 | mean_axis0 | 0.37 | 1.37 | +1.00 | 1.26 | +0.89 |
| (512, 512) | float32 | mean_axis1 | 3.07 | 3.33 | +0.26 | 3.24 | +0.18 |
| (512, 512) | float32 | scale | 0.67 | 0.64 | −0.03 | 0.40 | −0.26 |
| (512, 512) | float32 | sum | 5.00 | 2.93 | −2.07 | 2.73 | −2.27 |
| (512, 512) | float64 | broadcast_add | 1.83 | 1.63 | −0.20 | 0.77 | −1.06 |
| (512, 512) | float64 | mean | 2.09 | 1.73 | −0.36 | 2.42 | +0.32 |
| (512, 512) | float64 | mean_axis0 | 0.49 | 0.91 | +0.42 | 1.42 | +0.93 |
| (512, 512) | float64 | mean_axis1 | 2.58 | 1.44 | −1.14 | 1.21 | −1.38 |
| (512, 512) | float64 | scale | 0.25 | 0.30 | +0.06 | 1.21 | +0.97 |
| (512, 512) | float64 | sum | 0.58 | 1.51 | +0.93 | 0.63 | +0.05 |
| (1024, 1024) | float32 | broadcast_add | 0.72 | 0.83 | +0.11 | 0.84 | +0.12 |
| (1024, 1024) | float32 | mean | 0.54 | 0.52 | −0.02 | 0.46 | −0.08 |
| (1024, 1024) | float32 | mean_axis0 | 0.84 | 0.60 | −0.23 | 1.25 | +0.41 |
| (1024, 1024) | float32 | mean_axis1 | 3.12 | 3.19 | +0.07 | 3.11 | −0.01 |
| (1024, 1024) | float32 | scale | 0.59 | 0.56 | −0.03 | 0.56 | −0.03 |
| (1024, 1024) | float32 | sum | 0.46 | 0.46 | +0.00 | 0.45 | −0.00 |
| (1024, 1024) | float64 | broadcast_add | 3.53 | 1.64 | −1.89 | 1.64 | −1.89 |
| (1024, 1024) | float64 | mean | 1.40 | 2.64 | +1.24 | 0.95 | −0.45 |
| (1024, 1024) | float64 | mean_axis0 | 0.49 | 0.64 | +0.16 | 1.06 | +0.57 |
| (1024, 1024) | float64 | mean_axis1 | 2.83 | 1.44 | −1.39 | 1.45 | −1.38 |
| (1024, 1024) | float64 | scale | 0.82 | 0.59 | −0.23 | 0.59 | −0.23 |
| (1024, 1024) | float64 | sum | 2.64 | 1.53 | −1.11 | 1.30 | −1.34 |
| (2048, 2048) | float32 | broadcast_add | 1.00 | 1.03 | +0.03 | 1.00 | +0.00 |
| (2048, 2048) | float32 | mean | 2.32 | 2.28 | −0.05 | 2.22 | −0.10 |
| (2048, 2048) | float32 | mean_axis0 | 0.50 | 0.66 | +0.16 | 0.60 | +0.10 |
| (2048, 2048) | float32 | mean_axis1 | 2.89 | 2.91 | +0.02 | 2.90 | +0.01 |
| (2048, 2048) | float32 | scale | 0.91 | 0.88 | −0.04 | 1.00 | +0.09 |
| (2048, 2048) | float32 | sum | 0.83 | 2.13 | +1.31 | 2.23 | +1.40 |
| (2048, 2048) | float64 | broadcast_add | 1.36 | 1.50 | +0.14 | 1.46 | +0.10 |
| (2048, 2048) | float64 | mean | 1.38 | 1.05 | −0.32 | 1.01 | −0.37 |
| (2048, 2048) | float64 | mean_axis0 | 0.29 | 0.31 | +0.02 | 0.40 | +0.11 |
| (2048, 2048) | float64 | mean_axis1 | 2.24 | 1.23 | −1.02 | 0.85 | −1.39 |
| (2048, 2048) | float64 | scale | 1.14 | 0.84 | −0.30 | 0.85 | −0.29 |
| (2048, 2048) | float64 | sum | 1.81 | 1.22 | −0.59 | 1.09 | −0.72 |

## Run 20251117-103853 vs Baseline

Focused run after the latest fixes (`benchmarks/docker_results/20251117-103853`):

| Shape | Dtype | Op | Baseline Speedup | 20251117-103853 | Δ vs Base |
|-------|-------|----|------------------|-----------------|-----------|
| (512, 512) | float32 | sum | 5.00 | 3.64 | −1.36 |
| (512, 512) | float32 | mean_axis0 | 0.37 | 1.42 | +1.05 |
| (512, 512) | float64 | mean_axis1 | 2.58 | 1.58 | −1.01 |
| (1024, 1024) | float64 | mean_axis0 | 0.49 | 0.29 | −0.20 |
| (1024, 1024) | float32 | mean | 0.54 | 2.83 | +2.29 |
| (2048, 2048) | float64 | mean_axis1 | 2.24 | 1.00 | −1.25 |

Highlights:

- **Small-shape means improved:** new `SMALL_*` thresholds give solid gains for 512²/1024² mean/mean_axis0 (up to +2.3× speedup).
- **Row-wise regressions remain:** 512²/1024² float64 `mean_axis1` and 512² float32/float64 `sum` are still below the 11/16 baseline, so further SIMD tuning is needed.
- **Large mean_axis0 gaps persist:** 2048² float32/float64 `mean_axis0` stays below 1.0×; BLAS single-thread env forcing improved stability but not absolute speed.

## Notable Regressions

- **Small-shape sums and row means dropped sharply:** 512² float32 sum went from **5.0×** to **≈2.7–2.9×**, and 512² float64/1024² float64 mean_axis1 both lost >1.3× speedup, pointing to new overhead in row-reduction kernels.
- **1024² float64 broadcast_add regressed from 3.5× to 1.6×**, suggesting the latest routing or buffering changes hurt SIMD reuse.
- **Large float64 reductions slipped:** 2048² float64 mean, mean_axis1, and sum all fell by ≥0.3×, even though the BLAS backend stayed constant; we may have introduced extra synchronization or changed threading defaults.

## Improvements

- **Small float32 means now excel:** 512² float32 mean/mean_axis0 jumped by +2.35× and +1.0× speedup respectively, validating the SIMD tiled tweaks.
- **2048² float32 sum more than doubled (0.83× → 2.23×)**, and mean_axis0 crept upward (+0.10–0.16×), showing progress on the column kernel.
- **1024² float64 mean improved to 2.64×** in the first run (though the second run regressed, implying instability we need to chase).

## Next Steps

1. Investigate the highlighted regressions via perf/py-spy on Linux (focus on row reductions, broadcast_add, and BLAS-backed float64 operations).
2. Patch or revert the offending kernels/thresholds, then re-run Docker benchmarks to confirm recovery.
3. Continue the mean_axis0 catch-up experiments until all tracked sizes exceed NumPy.

