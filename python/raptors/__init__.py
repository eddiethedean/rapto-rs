"""User-facing convenience layer for the Rust-backed Raptors array core."""

from __future__ import annotations

from typing import Iterable

from . import _raptors as _core

RustArray = _core.RustArray

__all__ = [
    "RustArray",
    "array",
    "zeros",
    "ones",
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


def zeros(length: int) -> RustArray:
    """Return an array filled with zeros of the requested length."""
    return _core.zeros(length)


def ones(length: int) -> RustArray:
    """Return an array filled with ones of the requested length."""
    return _core.ones(length)


def from_numpy(ndarray) -> RustArray:
    """Create a Raptors array by copying data from a 1-D NumPy array."""
    return _core.from_numpy(ndarray)


def to_numpy(array: RustArray):
    """Convert a Raptors array into a NumPy array view."""
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "NumPy is required to convert Raptors arrays back to NumPy."
        ) from exc

    return _np.asarray(array.to_list(), dtype=_np.float64)

