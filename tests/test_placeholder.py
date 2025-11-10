import math
import subprocess
import sys

import pytest

import raptors


def as_nested_list(array):
    shape = array.shape
    data = array.to_list()
    if len(shape) == 1:
        return data
    if len(shape) == 2:
        rows, cols = shape
        return [data[row * cols : (row + 1) * cols] for row in range(rows)]
    raise NotImplementedError("tests currently support up to 2-D arrays")


def _numpy_is_usable() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import numpy"],
            check=False,
            capture_output=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def require_numpy():
    if not _NUMPY_USABLE:
        pytest.skip("NumPy is unavailable or crashes on import in this environment")
    import numpy as np  # type: ignore[import-not-found]

    return np


_NUMPY_USABLE = _numpy_is_usable()


def test_array_from_iterable():
    arr = raptors.array([1, 2, 3])
    assert isinstance(arr, raptors.RustArray)
    assert arr.to_list() == [1.0, 2.0, 3.0]
    assert len(arr) == 3


def test_zeros_and_ones_helpers():
    zeros = raptors.zeros(4)
    ones = raptors.ones(4)
    assert zeros.to_list() == [0.0, 0.0, 0.0, 0.0]
    assert ones.to_list() == [1.0, 1.0, 1.0, 1.0]


def test_elementwise_addition_and_length_validation():
    lhs = raptors.array([1.0, 2.0, 3.0])
    rhs = raptors.array([4.0, 5.0, 6.0])
    result = lhs.add(rhs)
    assert result.to_list() == [5.0, 7.0, 9.0]

    broadcasted = lhs.add(raptors.array([1.0]))
    assert broadcasted.to_list() == [2.0, 3.0, 4.0]

    with pytest.raises(ValueError):
        lhs.add(raptors.array([1.0, 2.0]))


def test_sum_and_mean():
    arr = raptors.array([2.0, 4.0, 6.0, 8.0])
    assert math.isclose(arr.sum(), 20.0)
    assert math.isclose(arr.mean(), 5.0)

    empty = raptors.array([])
    with pytest.raises(ValueError):
        empty.mean()


def test_array2d_shape_and_helpers():
    arr = raptors.array2d([[1, 2, 3], [4, 5, 6]])
    assert arr.shape == [2, 3]
    assert arr.ndim == 2
    assert len(arr) == 2

    zeros2d = raptors.zeros((2, 2))
    ones2d = raptors.ones((2, 3))
    assert zeros2d.shape == [2, 2]
    assert ones2d.shape == [2, 3]
    assert zeros2d.to_list() == [0.0, 0.0, 0.0, 0.0]
    assert ones2d.to_list() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_axis_reductions():
    arr = raptors.array2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    col_sum = arr.sum_axis(0)
    row_sum = arr.sum_axis(1)
    assert col_sum.shape == [3]
    assert row_sum.shape == [2]
    assert col_sum.to_list() == [5.0, 7.0, 9.0]
    assert row_sum.to_list() == [6.0, 15.0]

    col_mean = arr.mean_axis(0)
    row_mean = arr.mean_axis(1)
    assert col_mean.to_list() == [2.5, 3.5, 4.5]
    assert row_mean.to_list() == [2.0, 5.0]

    with pytest.raises(ValueError):
        arr.sum_axis(2)


def test_broadcast_add():
    row_vec = raptors.array([1.0, 2.0, 3.0])
    matrix = raptors.array2d([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    result = raptors.broadcast_add(matrix, row_vec)
    assert result.shape == [2, 3]
    assert result.to_list() == [
        11.0,
        22.0,
        33.0,
        41.0,
        52.0,
        63.0,
    ]

    col_vec = raptors.array2d([[100.0], [200.0]])
    result2 = raptors.broadcast_add(col_vec, matrix)
    assert result2.shape == [2, 3]
    assert result2.to_list() == [
        110.0,
        120.0,
        130.0,
        240.0,
        250.0,
        260.0,
    ]

    with pytest.raises(ValueError):
        raptors.broadcast_add(
            raptors.array2d([[1.0, 2.0], [3.0, 4.0]]),
            raptors.array([1.0, 2.0, 3.0]),
        )


def test_numpy_roundtrip():
    np = require_numpy()

    source = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    arr = raptors.from_numpy(source)
    assert arr.to_list() == [0.5, 1.5, 2.5]

    back_to_numpy = raptors.to_numpy(arr)
    assert back_to_numpy.dtype == np.float64
    assert back_to_numpy.shape == (3,)
    assert np.allclose(back_to_numpy, source)


def test_numpy_roundtrip_2d():
    np = require_numpy()

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr = raptors.from_numpy(matrix)
    assert arr.shape == [2, 2]
    assert arr.to_list() == [1.0, 2.0, 3.0, 4.0]

    back = raptors.to_numpy(arr)
    assert back.shape == (2, 2)
    assert np.allclose(back, matrix)


def test_float32_arrays():
    np = require_numpy()

    arr = raptors.array([1, 2, 3], dtype="float32")
    assert isinstance(arr, raptors.RustArrayF32)
    assert [round(v, 6) for v in arr.to_list()] == [1.0, 2.0, 3.0]

    zeros = raptors.zeros((2, 2), dtype="float32")
    assert isinstance(zeros, raptors.RustArrayF32)
    assert zeros.shape == [2, 2]

    back = raptors.to_numpy(arr)
    assert back.dtype == np.float32
    assert np.allclose(back, np.array([1, 2, 3], dtype=np.float32))


def test_int32_arrays_and_scale():
    arr = raptors.array([1, 2, 3], dtype="int32")
    assert isinstance(arr, raptors.RustArrayI32)
    assert arr.to_list() == [1, 2, 3]

    scaled = arr.scale(2.0)
    assert scaled.to_list() == [2, 4, 6]

    with pytest.raises(ValueError):
        arr.scale(0.5)


def test_int32_means_raise_for_fractional_results():
    arr = raptors.array([1, 2], dtype="int32")
    assert math.isclose(arr.mean(), 1.5)
    with pytest.raises(ValueError):
        arr.mean_axis(0)


def test_broadcast_add_float32():
    lhs = raptors.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    rhs = raptors.array([10.0, 20.0], dtype="float32")
    out = raptors.broadcast_add(lhs, rhs)
    assert isinstance(out, raptors.RustArrayF32)
    assert out.shape == [2, 2]
    assert [round(v, 3) for v in out.to_list()] == [11.0, 22.0, 13.0, 24.0]


def test_from_numpy_dtype_dispatch():
    np = require_numpy()

    f32_matrix = np.array([[1.0, 2.0]], dtype=np.float32)
    i32_matrix = np.array([[1, 2, 3]], dtype=np.int32)

    arr_f32 = raptors.from_numpy(f32_matrix)
    assert isinstance(arr_f32, raptors.RustArrayF32)
    arr_i32 = raptors.from_numpy(i32_matrix)
    assert isinstance(arr_i32, raptors.RustArrayI32)

    with pytest.raises(TypeError):
        raptors.from_numpy(np.array([1, 2, 3], dtype=np.int64))


def test_to_numpy_zero_copy_shares_buffer():
    np = require_numpy()

    arr = raptors.array([1.0, 2.0, 3.0])
    np_view = raptors.to_numpy(arr)
    assert np_view.base is arr

    np_view[1] = 42.5
    assert arr.to_list() == [1.0, 42.5, 3.0]


def test_from_numpy_zero_copy_shares_buffer():
    np = require_numpy()

    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr = raptors.from_numpy(base)
    assert arr.shape == [2, 2]

    base[0, 0] = 99.0
    assert arr.to_list()[0] == 99.0


def test_from_numpy_non_contiguous_falls_back_to_copy():
    np = require_numpy()

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]  # non-contiguous slice
    arr = raptors.from_numpy(view)
    assert arr.shape == [3, 2]

    view[0, 0] = -123.0
    assert arr.to_list()[0] != -123.0


def test_getitem_scalar_and_slice_1d():
    arr = raptors.array([10.0, 20.0, 30.0, 40.0])

    assert arr[0] == 10.0
    assert arr[-1] == 40.0

    tail = arr[1:]
    assert isinstance(tail, raptors.RustArray)
    assert tail.shape == [3]
    assert tail.to_list() == [20.0, 30.0, 40.0]

    reversed_arr = arr[::-1]
    assert reversed_arr.to_list() == [40.0, 30.0, 20.0, 10.0]

    with pytest.raises(IndexError):
        _ = arr[4]

    with pytest.raises(ValueError):
        _ = arr[::0]


def test_getitem_tuple_and_slicing_2d():
    arr = raptors.array2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert arr[0, 0] == 1.0
    assert arr[-1, -1] == 9.0

    first_col = arr[:, 0]
    assert isinstance(first_col, raptors.RustArray)
    assert first_col.shape == [3]
    assert first_col.to_list() == [1.0, 4.0, 7.0]

    lower_block = arr[1:, 1:]
    assert lower_block.shape == [2, 2]
    assert as_nested_list(lower_block) == [[5.0, 6.0], [8.0, 9.0]]

    row_slice = arr[1, :2]
    assert isinstance(row_slice, raptors.RustArray)
    assert row_slice.to_list() == [4.0, 5.0]

    with pytest.raises(IndexError):
        _ = arr[3, 0]

    with pytest.raises(IndexError):
        _ = arr[0, 3]

    with pytest.raises(IndexError):
        _ = arr[0, 0, 0]


def test_getitem_preserves_dtype():
    arr_i32 = raptors.array([1, 2, 3, 4], dtype="int32")
    subset = arr_i32[1:3]
    assert isinstance(subset, raptors.RustArrayI32)
    assert subset.to_list() == [2, 3]

    mat_f32 = raptors.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    col = mat_f32[:, 1]
    assert isinstance(col, raptors.RustArrayF32)
    assert [round(v, 6) for v in col.to_list()] == [2.0, 4.0]


def test_slice_with_reverse_and_step():
    arr = raptors.array2d(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
    )
    block = arr[::-1, ::2]
    assert block.shape == [3, 2]
    assert as_nested_list(block) == [[8.0, 10.0], [4.0, 6.0], [0.0, 2.0]]


def test_slice_array_helper_matches_getitem():
    arr = raptors.array([5.0, 6.0, 7.0])
    helper_slice = raptors.slice_array(arr, slice(1, None))
    assert helper_slice.to_list() == [6.0, 7.0]

    matrix = raptors.array2d([[1.0, 2.0], [3.0, 4.0]])
    helper_block = raptors.slice_array(matrix, (slice(None, None, -1), 1))
    assert helper_block.to_list() == [4.0, 2.0]


def test_index_array_helper_requires_integers():
    arr = raptors.array2d([[1.0, 2.0], [3.0, 4.0]])
    assert raptors.index_array(arr, 0, 1) == 2.0

    with pytest.raises(TypeError):
        raptors.index_array(arr, slice(None))

    with pytest.raises(TypeError):
        raptors.index_array(arr, 0, slice(None))


def test_simd_row_broadcast_add():
    np = require_numpy()

    matrix = raptors.from_numpy(np.arange(12, dtype=np.float64).reshape(3, 4))
    row = raptors.from_numpy(np.array([10.0, 20.0, 30.0, 40.0]))
    out = matrix.add(row)

    assert out.shape == [3, 4]
    expected = (np.arange(12, dtype=np.float64).reshape(3, 4) + np.array([10.0, 20.0, 30.0, 40.0])).ravel()
    assert out.to_list() == pytest.approx(expected.tolist())


def test_simd_same_shape_add():
    np = require_numpy()

    lhs = raptors.from_numpy(np.ones((2, 3), dtype=np.float64))
    rhs = raptors.from_numpy(np.full((2, 3), 5.0, dtype=np.float64))
    out = lhs.add(rhs)

    assert out.shape == [2, 3]
    assert out.to_list() == [6.0] * 6


def test_simd_column_broadcast_add():
    np = require_numpy()

    matrix = raptors.from_numpy(np.arange(12, dtype=np.float64).reshape(3, 4))
    column = raptors.from_numpy(np.array([[1.0], [2.0], [3.0]], dtype=np.float64))
    out = matrix.add(column)

    expected = (np.arange(12, dtype=np.float64).reshape(3, 4)
                + np.array([[1.0], [2.0], [3.0]], dtype=np.float64)).ravel()
    assert out.to_list() == pytest.approx(expected.tolist())


def test_simd_scale_matches_numpy():
    np = require_numpy()

    matrix = raptors.from_numpy(np.arange(6, dtype=np.float64).reshape(2, 3))
    scaled = matrix.scale(1.5)

    expected = (np.arange(6, dtype=np.float64).reshape(2, 3) * 1.5).ravel()
    assert scaled.to_list() == pytest.approx(expected.tolist())

