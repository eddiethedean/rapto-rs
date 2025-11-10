"""User-facing convenience layer for the Rust-backed Raptors array core."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

from . import _raptors as _core

RustArray = _core.RustArray
RustArrayF32 = _core.RustArrayF32
RustArrayI32 = _core.RustArrayI32
simd_enabled = _core.simd_enabled
ShapeLike = Union[int, Sequence[int]]
ArrayLike = Union[RustArray, RustArrayF32, RustArrayI32]

_SUPPORTED_ARRAY_TYPES: Tuple[type, ...] = (RustArray, RustArrayF32, RustArrayI32)
_ARRAY_CONSTRUCTORS = {
    RustArray: _core.array,
    RustArrayF32: _core.array_f32,
    RustArrayI32: _core.array_i32,
}

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
    "simd_enabled",
    "slice_array",
    "index_array",
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
    """Create a Raptors array from a NumPy array.

    Dtype is inferred from the NumPy array (float64, float32, int32).
    Returns a zero-copy view when the input is C-contiguous; otherwise
    a defensive copy is created.
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
    """Convert a Raptors array into a NumPy view of the matching dtype.

    Shares the underlying buffer when possible; falls back to a copy if
    a view cannot be exposed safely.
    """

    np = _ensure_numpy()
    if isinstance(array, _SUPPORTED_ARRAY_TYPES):
        return np.asarray(array)
    raise TypeError("expected a Raptors array instance")


def _ensure_numpy():
    try:
        import numpy as _np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "NumPy is required for Raptors <-> NumPy conversions."
        ) from exc
    return _np


def slice_array(array: ArrayLike, key):
    """Return a slice of a Raptors array using familiar Python indexing semantics."""

    return _slice_array_impl(array, key)


def index_array(array: ArrayLike, *indices: int):
    """Return a scalar value from a Raptors array using positional indices."""

    if not indices:
        raise TypeError("index_array requires at least one index")
    if any(not isinstance(idx, int) for idx in indices):
        raise TypeError("index_array only accepts integer indices")

    key = indices if len(indices) > 1 else indices[0]
    result = _slice_array_impl(array, key)
    if isinstance(result, _SUPPORTED_ARRAY_TYPES):
        raise TypeError("index_array expected scalar indices but received a slice")
    return result


def _slice_array_impl(array: ArrayLike, key):
    if not isinstance(array, _SUPPORTED_ARRAY_TYPES):
        raise TypeError("expected a Raptors array instance")

    shape = array.shape
    ndim = len(shape)
    if ndim == 0:
        raise TypeError("cannot slice scalar Raptors arrays")

    normalized_key = _expand_key(key, ndim)

    if ndim == 1:
        selector = _prepare_selector(normalized_key[0], shape[0])
        return _slice_1d(array, selector)

    if ndim == 2:
        row_selector = _prepare_selector(normalized_key[0], shape[0])
        col_selector = _prepare_selector(normalized_key[1], shape[1])
        return _slice_2d(array, row_selector, col_selector)

    raise NotImplementedError("slicing currently supports up to 2-D arrays")


def _expand_key(key, ndim: int) -> Tuple[object, ...]:
    if key is Ellipsis:
        return tuple(slice(None) for _ in range(ndim))

    if not isinstance(key, tuple):
        key = (key,)

    expanded: List[object] = []
    ellipsis_seen = False

    for position, component in enumerate(key):
        if component is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis")
            ellipsis_seen = True
            remaining = ndim - (len(key) - (position + 1)) - len(expanded)
            if remaining < 0:
                raise IndexError("too many indices for array")
            expanded.extend(slice(None) for _ in range(remaining))
        else:
            expanded.append(component)

    if len(expanded) > ndim:
        raise IndexError("too many indices for array")

    if len(expanded) < ndim:
        expanded.extend(slice(None) for _ in range(ndim - len(expanded)))

    return tuple(expanded)


def _prepare_selector(component, length: int):
    if isinstance(component, slice):
        try:
            start, stop, step = component.indices(length)
        except ValueError as exc:
            raise ValueError("slice step cannot be zero") from exc
        indices = list(range(start, stop, step))
        return ("slice", indices)

    if isinstance(component, int):
        return ("index", _normalize_index(component, length))

    raise TypeError("indices must be integers or slices")


def _normalize_index(index: int, length: int) -> int:
    if length == 0:
        raise IndexError("cannot index an empty axis")

    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise IndexError("index out of bounds")
    return index


def _slice_1d(array: ArrayLike, selector):
    data = array.to_list()

    kind, payload = selector
    if kind == "index":
        return data[payload]

    values = [data[idx] for idx in payload]
    return _construct_from_python(array, values)


def _slice_2d(array: ArrayLike, row_selector, col_selector):
    rows, cols = array.shape
    flat = array.to_list()
    matrix = _reshape_to_matrix(flat, rows, cols)

    row_kind, row_payload = row_selector
    col_kind, col_payload = col_selector

    if row_kind == "index":
        row = matrix[row_payload] if rows else []
        if col_kind == "index":
            return row[col_payload]
        values = [row[idx] for idx in col_payload]
        return _construct_from_python(array, values)

    selected_rows = [matrix[idx] for idx in row_payload]

    if col_kind == "index":
        values = [row[col_payload] for row in selected_rows]
        return _construct_from_python(array, values)

    values = [[row[idx] for idx in col_payload] for row in selected_rows]
    return _construct_from_python(array, values)


def _reshape_to_matrix(flat: List[Union[int, float]], rows: int, cols: int):
    if rows == 0:
        return []
    return [flat[row * cols : (row + 1) * cols] for row in range(rows)]


def _construct_from_python(template: ArrayLike, values):
    constructor = _ARRAY_CONSTRUCTORS.get(type(template))
    if constructor is None:
        raise TypeError("unsupported Raptors array type")
    return constructor(values)


def _bind_python_getitem():
    def __getitem__(self, key):
        return slice_array(self, key)

    for cls in _SUPPORTED_ARRAY_TYPES:
        setattr(cls, "__getitem__", __getitem__)


_bind_python_getitem()

