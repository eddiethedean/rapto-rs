#!/usr/bin/env python3
"""Validate persisted benchmark result artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_OPERATION_FIELDS = (
    "name",
    "numpy_mean_s",
    "raptors_mean_s",
    "speedup",
)


def iter_cases(payload: object) -> Iterable[Dict[str, object]]:
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        cases = payload.get("cases")
        if isinstance(cases, list):
            yield from cases
        else:
            raise ValueError("Expected 'cases' list inside JSON object")
    else:
        raise ValueError("Unexpected JSON payload type")


def validate_case(case: Dict[str, object], source: Path) -> None:
    shape = case.get("shape")
    dtype = case.get("dtype")
    operations = case.get("operations")

    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"{source}: invalid shape field {shape!r}")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{source}: invalid dtype field {dtype!r}")
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"{source}: missing operations for shape={shape}, dtype={dtype}")

    for op in operations:
        if not isinstance(op, dict):
            raise ValueError(f"{source}: operation entry is not an object")
        for field in REQUIRED_OPERATION_FIELDS:
            if field not in op:
                raise ValueError(f"{source}: operation missing '{field}' field")
        numpy_mean = op["numpy_mean_s"]
        raptors_mean = op["raptors_mean_s"]
        if not isinstance(numpy_mean, (int, float)) or numpy_mean < 0.0:
            raise ValueError(f"{source}: invalid numpy_mean_s {numpy_mean!r}")
        if not isinstance(raptors_mean, (int, float)) or raptors_mean < 0.0:
            raise ValueError(f"{source}: invalid raptors_mean_s {raptors_mean!r}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "benchmarks" / "results"
    if not results_dir.exists():
        print(f"[validate] results directory {results_dir} not found", file=sys.stderr)
        return 1

    json_files: List[Path] = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"[validate] no JSON result files found in {results_dir}", file=sys.stderr)
        return 1

    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: failed to parse JSON ({exc})")
        for case in iter_cases(payload):
            validate_case(case, path)

    print(f"[validate] checked {len(json_files)} result file(s) in {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


