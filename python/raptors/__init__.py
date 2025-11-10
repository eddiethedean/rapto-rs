"""User-facing convenience layer for the Rust-backed Raptors array core."""

from __future__ import annotations

from typing import Iterable, Sequence, Union

from . import _raptors as _core

RustArray = _core.RustArray
RustArrayF32 = _core.RustArrayF32
RustArrayI32 = _core.RustArrayI32
ShapeLike = Union[int, Sequence[int]]

__all__ = [
    "RustArray",
    "RustArrayF32",
    "RustArrayI32",
    "array",
    "array2d",
    "zeros",
    "ones",
    "broadcast_add",
    "from_numpy",
    "to_numpy",
    "__version__",
    "__author__",
    "__github__",
]

__version__ = getattr(_core, "__version__", "0.0.2")
__author__ = getattr(_core, "__author__", "Odos Matthews <odosmatthews@gmail.com>")
__github__ = getattr(_core, "__github__", "https://github.com/eddiethedean")


def array(values: Iterable[float], *, dtype: str = "float64"):
    """Construct a Raptors array from any Python iterable.

    Parameters
    ----------
    values:
        1-D or 2-D iterable of numeric values (nested iterables create 2-D arrays).
    dtype:
        Literal "float64", "float32", or "int32" (default "float64").
    """

    dtype = dtype.lower()
    if dtype == "float64":
        return _core.array(values)
    if dtype == "float32":
        return _core.array_f32(values)
    if dtype == "int32":
        return _core.array_i32(values)
    raise ValueError(f"unsupported dtype '{dtype}'")


def array2d(rows: Iterable[Iterable[float]], *, dtype: str = "float64"):
    """Construct a 2-D Raptors array from a nested iterable of numeric values."""

    return array(rows, dtype=dtype)


def zeros(shape: ShapeLike, *, dtype: str = "float64"):
    """Return an array of the requested shape filled with zeros."""

    dtype = dtype.lower()
    if dtype == "float64":
        return _core.zeros(shape)
    if dtype == "float32":
        return _core.zeros_f32(shape)
    if dtype == "int32":
        return _core.zeros_i32(shape)
    raise ValueError(f"unsupported dtype '{dtype}'")


def ones(shape: ShapeLike, *, dtype: str = "float64"):
    """Return an array of the requested shape filled with ones."""

    dtype = dtype.lower()
    if dtype == "float64":
        return _core.ones(shape)
    if dtype == "float32":
        return _core.ones_f32(shape)
    if dtype == "int32":
        return _core.ones_i32(shape)
    raise ValueError(f"unsupported dtype '{dtype}'")


def broadcast_add(lhs, rhs):
    """Return elementwise sum with NumPy-style broadcasting rules.

    Both inputs must be the same Raptors dtype.
    """

    if isinstance(lhs, RustArray) and isinstance(rhs, RustArray):
        return _core.broadcast_add(lhs, rhs)
    if isinstance(lhs, RustArrayF32) and isinstance(rhs, RustArrayF32):
        return _core.broadcast_add_f32(lhs, rhs)
    if isinstance(lhs, RustArrayI32) and isinstance(rhs, RustArrayI32):
        return _core.broadcast_add_i32(lhs, rhs)
    raise TypeError("broadcast_add requires both operands to share the same Raptors dtype")


def from_numpy(ndarray):
    """Create a Raptors array by copying data from a NumPy array.

    Dtype is inferred from the NumPy array (float64, float32, int32).
    """

    np = _ensure_numpy()
    dtype = ndarray.dtype
    if dtype == np.float64:
        return _core.from_numpy(ndarray)
    if dtype == np.float32:
        return _core.from_numpy_f32(ndarray)
    if dtype == np.int32:
        return _core.from_numpy_i32(ndarray)
    raise TypeError(
        "from_numpy currently supports float64, float32, and int32 NumPy arrays"
    )


def to_numpy(array):
    """Convert a Raptors array into a NumPy array of the matching dtype."""

    np = _ensure_numpy()
    if isinstance(array, RustArray):
        return np.asarray(array.to_numpy(), dtype=np.float64)
    if isinstance(array, RustArrayF32):
        return np.asarray(array.to_numpy(), dtype=np.float32)
    if isinstance(array, RustArrayI32):
        return np.asarray(array.to_numpy(), dtype=np.int32)
    raise TypeError("expected a Raptors array instance")


def _ensure_numpy():
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "NumPy is required for Raptors <-> NumPy conversions."
        ) from exc
    return _np

