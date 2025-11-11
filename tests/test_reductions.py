import numpy as np
import pytest

raptors = pytest.importorskip("raptors")


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((8, 16), 0),
        ((8, 16), 1),
        ((64, 128), 0),
        ((64, 128), 1),
        ((128, 64), 0),
        ((128, 64), 1),
        ((512, 512), 0),
        ((512, 512), 1),
        ((1024, 1024), 0),
        ((1024, 1024), 1),
        ((200, 200), 0),
        ((200, 200), 1),
    ],
)
def test_float64_mean_axes_match_numpy(shape, axis):
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(shape).astype(np.float64)
    rap = raptors.from_numpy(arr)
    expected = arr.mean(axis=axis)
    result = rap.mean_axis(axis).to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((8, 16), 0),
        ((8, 16), 1),
        ((64, 128), 0),
        ((64, 128), 1),
        ((128, 64), 0),
        ((128, 64), 1),
        ((512, 512), 0),
        ((512, 512), 1),
        ((1024, 1024), 0),
        ((1024, 1024), 1),
    ],
)
def test_float32_mean_axes_match_numpy(shape, axis):
    rng = np.random.default_rng(24)
    arr = rng.standard_normal(shape).astype(np.float32)
    rap = raptors.from_numpy(arr)
    expected = arr.mean(axis=axis)
    result = rap.mean_axis(axis).to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (128, 64)])
def test_column_broadcast_matches_numpy(shape):
    rng = np.random.default_rng(7)
    arr = rng.standard_normal(shape).astype(np.float32)
    col = rng.standard_normal(shape[0]).astype(np.float32)
    rap_arr = raptors.from_numpy(arr)
    rap_col = raptors.from_numpy(col)
    expected = arr + col[:, None]
    result = raptors.broadcast_add(rap_arr, rap_col).to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(8, 16), (64, 128), (128, 64), (200, 200)])
def test_scale_matches_numpy(shape):
    rng = np.random.default_rng(13)
    arr = rng.standard_normal(shape).astype(np.float64)
    factor = 1.2345
    rap_arr = raptors.from_numpy(arr)
    expected = arr * factor
    result = rap_arr.scale(factor).to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_float64_axis0_small_matrix_stack_fast_path():
    rng = np.random.default_rng(314)
    arr = rng.standard_normal((200, 180)).astype(np.float64)
    rap = raptors.from_numpy(arr)
    expected = arr.mean(axis=0)
    result = rap.mean_axis(0).to_numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
