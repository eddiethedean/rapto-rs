#!/usr/bin/env python3
"""Compare the performance of Raptors vs NumPy for common array operations."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - runtime check only
    raise SystemExit("NumPy is required to run this benchmark script") from exc

NUMPY_DTYPES = {
    "float64": np.float64,
    "float32": np.float32,
    "int32": np.int32,
}

RAPTORS_DTYPES = {
    "float64": "float64",
    "float32": "float32",
    "int32": "int32",
}

PRESET_SUITES = {
    "2d": (
        {"shape": (512, 512), "dtype": "float64"},
        {"shape": (512, 512), "dtype": "float32"},
        {"shape": (1024, 1024), "dtype": "float64"},
        {"shape": (1024, 1024), "dtype": "float32"},
        {"shape": (2048, 2048), "dtype": "float64"},
        {"shape": (2048, 2048), "dtype": "float32"},
    ),
    "mixed": (
        {"shape": (4096,), "dtype": "float64"},
        {"shape": (2048,), "dtype": "float32"},
        {"shape": (512, 512), "dtype": "int32"},
    ),
}

OperationFn = Callable[[], None]
OperationPair = Tuple[OperationFn, OperationFn, str]


def collect_numpy_config() -> Dict[str, object]:
    """Return NumPy configuration/BLAS information for metadata/logging."""
    from numpy import __config__ as np_config  # local import to avoid unused in tests

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np_config.show()
    show_text = capture.getvalue().strip()

    info = {}
    try:
        blas_info = np_config.get_info("blas_opt_info")
    except Exception:  # pragma: no cover - defensive
        blas_info = {}
    try:
        lapack_info = np_config.get_info("lapack_opt_info")
    except Exception:  # pragma: no cover - defensive
        lapack_info = {}

    info["show"] = show_text
    info["blas_opt_info"] = blas_info
    info["lapack_opt_info"] = lapack_info
    return info


def parse_shape(value: str) -> Tuple[int, ...]:
    parts = value.lower().replace("*", "x").split("x")
    dims: List[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            dims.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid shape component '{part}'") from exc
    if not dims:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    return tuple(dims)


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def benchmark(fn: OperationFn, warmup: int, repeats: int) -> List[float]:
    for _ in range(max(warmup, 0)):
        fn()
    timings: List[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def build_operations(
    np_base: np.ndarray,
    raptors_base,
    raptors_mod,
    dtype: str,
) -> Dict[str, OperationPair]:
    ops: Dict[str, OperationPair] = {}

    # Sum of all elements
    ops["sum"] = (
        lambda: (_ := float(np_base.sum())),
        lambda: (_ := raptors_base.sum()),
        "Total sum",
    )

    # Mean of all elements
    ops["mean"] = (
        lambda: (_ := float(np_base.mean())),
        lambda: (_ := raptors_base.mean()),
        "Mean of all elements",
    )

    if np_base.ndim >= 1:
        ops["mean_axis0"] = (
            lambda: (_ := np_base.mean(axis=0)),
        lambda: (_ := raptors_base.mean_axis(0)),
            "Mean across axis 0",
        )

    if np_base.ndim >= 2:
        ops["mean_axis1"] = (
            lambda: (_ := np_base.mean(axis=1)),
        lambda: (_ := raptors_base.mean_axis(1)),
            "Mean across axis 1",
        )

    # Broadcasting addition against a 1-D vector or scalar.
    if np_base.ndim == 1:
        rhs_np = np.array([1], dtype=NUMPY_DTYPES[dtype])
    else:
        rhs_np = np.arange(np_base.shape[-1], dtype=NUMPY_DTYPES[dtype])
    rhs_r = raptors_mod.from_numpy(rhs_np)
    ops["broadcast_add"] = (
        lambda: (_ := np_base + rhs_np),
        lambda: (_ := raptors_mod.broadcast_add(raptors_base, rhs_r)),
        "Broadcast addition",
    )

    # Scaling (supports integer or float factors as appropriate).
    scale_factor = 2 if dtype == "int32" else 1.0009765625
    ops["scale"] = (
        lambda: (_ := np_base * scale_factor),
        lambda: (_ := raptors_base.scale(float(scale_factor))),
        "Scale array",
    )

    return ops


def choose_operations(
    available: Dict[str, OperationPair],
    requested: Iterable[str],
) -> Dict[str, OperationPair]:
    operations: Dict[str, OperationPair] = {}
    for name in requested:
        key = name.lower()
        if key not in available:
            print(f"[warn] Operation '{name}' is not available for the current shape; skipping.")
            continue
        operations[key] = available[key]
    if not operations:
        raise SystemExit("No valid operations selected; nothing to benchmark.")
    return operations


def summarize(name: str, times: List[float]) -> Tuple[float, float]:
    if len(times) >= 5:
        sorted_times = sorted(times)
        trimmed = sorted_times[1:-1]
    else:
        trimmed = times
    mean = statistics.mean(trimmed)
    std = statistics.stdev(trimmed) if len(trimmed) > 1 else 0.0
    return mean, std


def format_ms(seconds: float) -> float:
    return seconds * 1_000.0


def run_case(
    raptors_mod,
    shape: Tuple[int, ...],
    dtype_name: str,
    operations: Iterable[str] | None,
    warmup: int,
    repeats: int,
    layout: str,
) -> Dict[str, object]:
    np_dtype = NUMPY_DTYPES[dtype_name]
    base_np = np.arange(product(shape), dtype=np_dtype).reshape(shape)
    if layout == "transpose" and base_np.ndim >= 2:
        base_np = base_np.T
    elif layout == "fortran":
        base_np = np.asfortranarray(base_np)
    else:
        base_np = np.ascontiguousarray(base_np)
    shape = tuple(int(dim) for dim in base_np.shape)
    total = int(base_np.size)

    print(f"Benchmarking shape={shape}, dtype={dtype_name}, elements={total:,}")
    print(f"Warm-up iterations={warmup}, timed repeats={repeats}, layout={layout}\n")

    base_raptors = raptors_mod.from_numpy(base_np)

    available_ops = build_operations(base_np, base_raptors, raptors_mod, dtype_name)
    op_names = (
        operations if operations else available_ops.keys()
    )
    selected_ops = choose_operations(available_ops, op_names)

    table_rows = []
    json_results: List[Dict[str, object]] = []

    header = f"{'Operation':<16}{'Description':<28}{'NumPy (ms)':>14}{'Raptors (ms)':>14}{'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for name, (np_fn, raptors_fn, description) in selected_ops.items():
        np_times = benchmark(np_fn, warmup, repeats)
        r_times = benchmark(raptors_fn, warmup, repeats)
        np_mean, np_std = summarize(name, np_times)
        r_mean, r_std = summarize(name, r_times)
        ratio = np_mean / r_mean if r_mean else float("inf")

        np_ms = format_ms(np_mean)
        np_sd = format_ms(np_std)
        r_ms = format_ms(r_mean)
        r_sd = format_ms(r_std)
        ratio_display = f"{ratio:.2f}x" if ratio != float("inf") else "inf"

        print(
            f"{name:<16}{description:<28}"
            f"{np_ms:>9.2f}±{np_sd:>4.2f}{r_ms:>9.2f}±{r_sd:>4.2f}{ratio_display:>10}"
        )

        table_rows.append((name, description, np_mean, np_std, r_mean, r_std, ratio))
        json_results.append(
            {
                "name": name,
                "description": description,
                "numpy_mean_s": np_mean,
                "numpy_std_s": np_std,
                "raptors_mean_s": r_mean,
                "raptors_std_s": r_std,
                "speedup": ratio,
            }
        )

    print()
    print("Speedup shows NumPy time divided by Raptors time (higher is better for Raptors).")

    return {
        "shape": list(shape),
        "dtype": dtype_name,
        "elements": total,
        "layout": layout,
        "operations": json_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        default="1024",
        type=parse_shape,
        help="Array shape (e.g. 1024 or 2048x2048)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(NUMPY_DTYPES.keys()),
        default="float64",
        help="Data type to benchmark",
    )
    parser.add_argument(
        "--operations",
        nargs="*",
        default=["sum", "mean", "mean_axis0", "mean_axis1", "broadcast_add", "scale"],
        help="Operations to benchmark (default: common set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warm-up iterations before timing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed iterations per operation",
    )
    parser.add_argument(
        "--simd-mode",
        choices=["auto", "force", "disable"],
        default="auto",
        help="Control RAPTORS_SIMD dispatch (auto: detect, force: enable, disable: scalar fallback)",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(PRESET_SUITES.keys()),
        help="Run a preset suite of shapes/dtypes (overrides --shape/--dtype).",
    )
    parser.add_argument(
        "--layout",
        choices=["contiguous", "transpose", "fortran"],
        default="contiguous",
        help="Arrange NumPy inputs to test contiguous, transposed, or Fortran-order layouts.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write detailed benchmark results in JSON format.",
    )
    parser.add_argument(
        "--validate-json",
        help="Optional baseline JSON; fail if Raptors timings exceed recorded limits.",
    )
    parser.add_argument(
        "--validate-slack",
        type=float,
        default=0.0,
        help="Additional slack (ms) added to each baseline max when validating.",
    )
    parser.add_argument(
        "--log-numpy-config",
        dest="log_numpy_config",
        action="store_true",
        help="Print NumPy build / BLAS configuration before running benchmarks.",
    )
    parser.add_argument(
        "--skip-numpy-config",
        dest="log_numpy_config",
        action="store_false",
        help="Suppress NumPy configuration logging.",
    )
    parser.set_defaults(log_numpy_config=True)

    args = parser.parse_args(argv)

    if args.simd_mode == "force":
        os.environ["RAPTORS_SIMD"] = "1"
    elif args.simd_mode == "disable":
        os.environ["RAPTORS_SIMD"] = "0"
    else:
        os.environ.pop("RAPTORS_SIMD", None)

    import raptors as raptors_mod  # type: ignore[import]

    numpy_metadata: Dict[str, object] | None = None
    if args.log_numpy_config:
        numpy_metadata = collect_numpy_config()
        print("== NumPy configuration ==")
        show_text = str(numpy_metadata.get("show", "")).strip()
        if show_text:
            print(show_text)
        else:
            print("(numpy.__config__.show() produced no output)")
        blas_info = numpy_metadata.get("blas_opt_info", {})
        if blas_info:
            print("\n[blas_opt_info]")
            for key, value in sorted(blas_info.items()):
                print(f"{key}: {value}")
        lapack_info = numpy_metadata.get("lapack_opt_info", {})
        if lapack_info:
            print("\n[lapack_opt_info]")
            for key, value in sorted(lapack_info.items()):
                print(f"{key}: {value}")
        print()

    cases: List[Dict[str, object]] = []
    if args.suite:
        suite_cases = PRESET_SUITES.get(args.suite)
        if not suite_cases:
            raise SystemExit(f"Unknown suite '{args.suite}'.")
        cases.extend(suite_cases)
    else:
        cases.append({"shape": args.shape, "dtype": args.dtype})

    json_output: List[Dict[str, object]] = []
    for index, case in enumerate(cases):
        shape = tuple(case["shape"])
        dtype_name = str(case["dtype"])
        if index > 0:
            print("\n" + "=" * 72 + "\n")
        result = run_case(
            raptors_mod=raptors_mod,
            shape=shape,
            dtype_name=dtype_name,
            operations=args.operations,
            warmup=args.warmup,
            repeats=args.repeats,
            layout=args.layout,
        )
        json_output.append(result)

    try:
        raptors_threading = raptors_mod.threading_info()  # type: ignore[attr-defined]
    except Exception:
        raptors_threading = None

    metadata = {
        "numpy_version": np.__version__,
        "numpy_config": numpy_metadata,
        "simd_mode": args.simd_mode,
        "layout": args.layout,
        "env": {
            "RAPTORS_THREADS": os.environ.get("RAPTORS_THREADS"),
            "RAPTORS_SIMD": os.environ.get("RAPTORS_SIMD"),
        },
        "timestamp": time.time(),
        "raptors_threading": raptors_threading,
        "raptors_simd_enabled": getattr(raptors_mod, "simd_enabled", lambda: None)(),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        payload = {"metadata": metadata, "cases": json_output}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote JSON results to {output_path}")

        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        print(f"Wrote NumPy/dispatch metadata to {metadata_path}")

    if args.validate_json:
        with open(args.validate_json, "r", encoding="utf-8") as handle:
            baselines = json.load(handle)

        failure_messages: List[str] = []
        # Build lookup for measured results
        measurements: Dict[Tuple[Tuple[int, ...], str, str], float] = {}
        for case in json_output:
            shape = tuple(int(dim) for dim in case["shape"])
            dtype_name = str(case["dtype"])
            for op in case["operations"]:
                key = (shape, dtype_name, str(op["name"]))
                measurements[key] = float(op["raptors_mean_s"]) * 1_000.0  # ms

        for entry in baselines:
            shape = tuple(int(dim) for dim in entry.get("shape", []))
            dtype_name = str(entry.get("dtype"))
            operation = str(entry.get("operation"))
            max_ms = float(entry.get("max_raptors_ms"))
            key = (shape, dtype_name, operation)
            measured = measurements.get(key)
            if measured is None:
                failure_messages.append(
                    f"[validate] Missing measurement for shape={shape}, dtype={dtype_name}, operation={operation}"
                )
                continue
            if measured > max_ms + args.validate_slack:
                failure_messages.append(
                    f"[validate] {operation} @ shape={shape} dtype={dtype_name}: {measured:.2f} ms > allowed {max_ms + args.validate_slack:.2f} ms"
                )

        if failure_messages:
            print("\nValidation failures detected:")
            for message in failure_messages:
                print(message)
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - command line entry point
    sys.exit(main())
