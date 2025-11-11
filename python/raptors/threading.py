from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import _raptors as _core


@dataclass(frozen=True)
class AdaptiveThreshold:
    dtype: str
    baseline_cutover: int
    recommended_cutover: Optional[int]
    median_elements_per_ms: float
    seq_median_elements_per_ms: float
    p95_elements_per_ms: Optional[float]
    seq_p95_elements_per_ms: Optional[float]
    variance_ratio: Optional[float]
    seq_variance_ratio: Optional[float]
    sample_count: int
    seq_sample_count: int
    samples: List[float]
    seq_samples: List[float]
    target_latency_ms: float


@dataclass(frozen=True)
class LastEvent:
    dtype: str
    elements: int
    duration_ms: float
    tiles_processed: int
    partial_buffer: int
    parallel: bool
    operation: str
    chunk_rows: Optional[int]
    chunk_len: Optional[int]
    chunk_count: Optional[int]
    pool_threads: Optional[int]
    strategy: str


@dataclass(frozen=True)
class ThreadingDiagnostics:
    parallel_min_elements: int
    baseline_cutovers: Dict[str, int]
    dimension_thresholds: Dict[str, List[int]]
    thread_pool: Optional[Dict[str, int]]
    adaptive_thresholds: Dict[str, AdaptiveThreshold]
    last_event: Optional[LastEvent]


def _coerce_int_map(payload) -> Dict[str, int]:
    return {str(key): int(value) for key, value in (payload or {}).items()}


def _coerce_pool(payload) -> Optional[Dict[str, int]]:
    if not isinstance(payload, dict):
        return None
    return {str(key): int(value) for key, value in payload.items()}


def _coerce_dim_map(payload) -> Dict[str, List[int]]:
    dim_map: Dict[str, List[int]] = {}
    for key, values in (payload or {}).items():
        coerced = [int(component) for component in values]
        dim_map[str(key)] = coerced
    return dim_map


def _build_adaptive_thresholds(payload) -> Dict[str, AdaptiveThreshold]:
    result: Dict[str, AdaptiveThreshold] = {}
    for dtype, details in (payload or {}).items():
        samples = [float(value) for value in details.get("samples", [])]
        seq_samples = [float(value) for value in details.get("seq_samples", [])]
        recommended = details.get("recommended_cutover")
        result[str(dtype)] = AdaptiveThreshold(
            dtype=str(dtype),
            baseline_cutover=int(details.get("baseline_cutover", 0)),
            recommended_cutover=int(recommended) if recommended is not None else None,
            median_elements_per_ms=float(details.get("median_elements_per_ms", 0.0)),
            seq_median_elements_per_ms=float(
                details.get("seq_median_elements_per_ms", 0.0)
            ),
            p95_elements_per_ms=(
                float(details["p95_elements_per_ms"])
                if details.get("p95_elements_per_ms") not in (None, "null")
                else None
            ),
            seq_p95_elements_per_ms=(
                float(details["seq_p95_elements_per_ms"])
                if details.get("seq_p95_elements_per_ms") not in (None, "null")
                else None
            ),
            variance_ratio=(
                float(details["variance_ratio"])
                if details.get("variance_ratio") not in (None, "null")
                else None
            ),
            seq_variance_ratio=(
                float(details["seq_variance_ratio"])
                if details.get("seq_variance_ratio") not in (None, "null")
                else None
            ),
            sample_count=int(details.get("sample_count", 0)),
            seq_sample_count=int(details.get("seq_sample_count", 0)),
            samples=samples,
            seq_samples=seq_samples,
            target_latency_ms=float(details.get("target_latency_ms", 0.0)),
        )
    return result


def _build_last_event(payload) -> Optional[LastEvent]:
    if not isinstance(payload, dict):
        return None

    def _opt_int(value) -> Optional[int]:
        if value in (None, "null"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return LastEvent(
        dtype=str(payload.get("dtype", "")),
        elements=int(payload.get("elements", 0)),
        duration_ms=float(payload.get("duration_ms", 0.0)),
        tiles_processed=int(payload.get("tiles_processed", 0)),
        partial_buffer=int(payload.get("partial_buffer", 0)),
        parallel=bool(payload.get("parallel", False)),
        operation=str(payload.get("operation", "")),
        chunk_rows=_opt_int(payload.get("chunk_rows")),
        chunk_len=_opt_int(payload.get("chunk_len")),
        chunk_count=_opt_int(payload.get("chunk_count")),
        pool_threads=_opt_int(payload.get("pool_threads")),
        strategy=str(payload.get("strategy", "")),
    )


def threading_info() -> ThreadingDiagnostics:
    """Return live diagnostics for Raptors' adaptive threading heuristics."""

    raw = _core.threading_info()
    return ThreadingDiagnostics(
        parallel_min_elements=int(raw.get("parallel_min_elements", 0)),
        baseline_cutovers=_coerce_int_map(raw.get("baseline_cutovers")),
        dimension_thresholds=_coerce_dim_map(raw.get("dimension_thresholds")),
        thread_pool=_coerce_pool(raw.get("thread_pool")),
        adaptive_thresholds=_build_adaptive_thresholds(raw.get("adaptive_thresholds")),
        last_event=_build_last_event(raw.get("last_event")),
    )

