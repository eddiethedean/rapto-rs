"""User-facing convenience layer for the Rust-backed Raptors array core."""

from __future__ import annotations

from typing import Iterable, Sequence, Union

from . import _raptors as _core

RustArray = _core.RustArray
ShapeLike = Union[int, Sequence[int]]

__all__ = [
    "RustArray",
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


def array(values: Iterable[float]) -> RustArray:
    """Construct a Raptors array from any Python iterable of numeric values."""
    return _core.array(values)


def array2d(rows: Iterable[Iterable[float]]) -> RustArray:
    """Construct a 2-D Raptors array from a nested iterable of numeric values."""
    return _core.array(rows)


def zeros(shape: ShapeLike) -> RustArray:
    """Return an array of the requested shape filled with zeros."""
    return _core.zeros(shape)


def ones(shape: ShapeLike) -> RustArray:
    """Return an array of the requested shape filled with ones."""
    return _core.ones(shape)


def broadcast_add(lhs: RustArray, rhs: RustArray) -> RustArray:
    """Return elementwise sum with NumPy-style broadcasting rules."""
    return lhs.add(rhs)


def from_numpy(ndarray) -> RustArray:
    """Create a Raptors array by copying data from a NumPy array."""
    return _core.from_numpy(ndarray)


def to_numpy(array: RustArray):
    """Convert a Raptors array into a NumPy array."""
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "NumPy is required to convert Raptors arrays back to NumPy."
        ) from exc

    return _np.asarray(array.to_numpy(), dtype=_np.float64)

