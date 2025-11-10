import math

import pytest

import raptors


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

    # Broadcasting with a scalar-like 1-element array works
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
    np = pytest.importorskip("numpy")

    source = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    arr = raptors.from_numpy(source)
    assert arr.to_list() == [0.5, 1.5, 2.5]

    back_to_numpy = raptors.to_numpy(arr)
    assert back_to_numpy.dtype == np.float64
    assert back_to_numpy.shape == (3,)
    assert np.allclose(back_to_numpy, source)


def test_numpy_roundtrip_2d():
    np = pytest.importorskip("numpy")

    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr = raptors.from_numpy(matrix)
    assert arr.shape == [2, 2]
    assert arr.to_list() == [1.0, 2.0, 3.0, 4.0]

    back = raptors.to_numpy(arr)
    assert back.shape == (2, 2)
    assert np.allclose(back, matrix)

