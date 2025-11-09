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

    with pytest.raises(ValueError):
        lhs.add(raptors.array([1.0]))


def test_sum_and_mean():
    arr = raptors.array([2.0, 4.0, 6.0, 8.0])
    assert math.isclose(arr.sum(), 20.0)
    assert math.isclose(arr.mean(), 5.0)

    empty = raptors.array([])
    with pytest.raises(ValueError):
        empty.mean()


def test_numpy_roundtrip():
    np = pytest.importorskip("numpy")

    source = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    arr = raptors.from_numpy(source)
    assert arr.to_list() == [0.5, 1.5, 2.5]

    back_to_numpy = raptors.to_numpy(arr)
    assert back_to_numpy.dtype == np.float64
    assert back_to_numpy.shape == (3,)
    assert np.allclose(back_to_numpy, source)

