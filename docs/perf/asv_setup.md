# Airspeed Velocity (ASV) Benchmark Workflow

## Prerequisites
- Python ≥3.10 and `pipx` (recommended).
- `asv` installed in an isolated environment:

```bash
pipx install asv
```

## Repository Layout

```
benchmarks/
  asv.conf.json
  raptors/
    bench_scale.py
    bench_reductions.py
    bench_broadcast.py
benchmarks/baselines/asv/
```

- `asv.conf.json` defines environments (CPython releases), benchmarks package, and result storage.
- Each `bench_*.py` exposes classes with `time_*` methods following NumPy’s suite conventions.
- Baselines captured with `asv run` live under `benchmarks/baselines/asv/` and are compared during CI validation.

## Running Benchmarks Locally

```bash
cd benchmarks
asv run --python=same --quick
asv run --python=3.11 --bench ScaleSuite.time_scale_2048x2048 --show-stderr
```

- `--quick` mirrors NumPy’s smoke runs (single sample) for PR validation.
- Full sweeps (`asv run`) are reserved for nightly jobs or release candidates.

## Recording Baselines

```bash
asv run --python=same --steps 5
asv publish
cp .asv/results/* benchmarks/baselines/asv/
```

- Commit the updated JSON files when performance improves.
- CI compares the most recent run against these baselines with optional slack configured in `scripts/validate_benchmark_results.py`.

## CI Integration

1. Add `ci/benchmarks.yml` that:
   - Sets up Python environments.
   - Runs `asv run --python=same --quick`.
   - Uploads `.asv/results` as an artifact.
2. Schedule a nightly run to capture full-step benchmarks and open an issue if regressions exceed baseline slack.

## Reporting

- `asv publish` emits HTML under `benchmarks/html`. Upload to GitHub Pages (similar to NumPy’s https://pv.github.io/numpy-bench/).
- Include a summary table in `docs/perf_report.md` after significant optimisations.

