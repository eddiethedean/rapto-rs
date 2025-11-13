from __future__ import annotations

import os
from typing import Sequence, Tuple, Union

import numpy as np
import raptors


NUMPY_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
}


def configure_env(threads: Union[int, str] = 1, simd_mode: str = "auto") -> None:
    if threads in (None, "auto"):
        os.environ.pop("RAPTORS_THREADS", None)
    else:
        os.environ["RAPTORS_THREADS"] = str(threads)

    if simd_mode == "auto":
        os.environ.pop("RAPTORS_SIMD", None)
    else:
        os.environ["RAPTORS_SIMD"] = {"force": "1", "disable": "0"}.get(simd_mode, simd_mode)


def make_inputs(
    shape: Tuple[int, ...],
    dtype: str,
    layout: str,
) -> Tuple[np.ndarray, raptors.RustArray]:
    np_dtype = NUMPY_DTYPES[dtype]
    base = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
    if layout == "transpose" and base.ndim >= 2:
        base = np.ascontiguousarray(base.T)
    elif layout == "fortran":
        base = np.asfortranarray(base)
    else:
        base = np.ascontiguousarray(base)
    raptor = raptors.from_numpy(base)
    return base, raptor


def scale_factor(dtype: str) -> float:
    return 1.0009765625 if dtype in ("float32", "float64") else 2.0

