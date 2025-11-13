from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


BENCH_CODE = """
import json
import numpy as np
import raptors

arr = raptors.array(np.linspace(0.0, 1.0, 64))
result = {
    "simd": raptors.simd_enabled(),
    "sum": arr.sum(),
    "mean": arr.mean(),
    "scale_first": arr.scale(2.0).to_list()[0],
}
print(json.dumps(result))
"""


SCALAR_BROADCAST_CODE = """
import numpy as np
import raptors

base = np.arange(48, dtype=np.float32).reshape(6, 8)
arr = base.T
col = np.linspace(0.5, 1.5, arr.shape[0], dtype=np.float32)
row = np.linspace(-1.0, 1.0, arr.shape[1], dtype=np.float32)

rap_arr = raptors.from_numpy(arr)
rap_col = raptors.from_numpy(col)
rap_row = raptors.from_numpy(row)

col_out = raptors.broadcast_add(rap_arr, rap_col).to_numpy()
row_out = raptors.broadcast_add(rap_arr, rap_row).to_numpy()

np.testing.assert_allclose(col_out, arr + col[:, None], rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(row_out, arr + row, rtol=1e-5, atol=1e-6)
"""


def run_helper(simd_env: str | None):
    env = os.environ.copy()
    if simd_env is None:
        env.pop("RAPTORS_SIMD", None)
    else:
        env["RAPTORS_SIMD"] = simd_env
    output = subprocess.check_output(
        [sys.executable, "-c", BENCH_CODE], env=env, text=True
    )
    return json.loads(output)


@pytest.mark.parametrize("flag,expected", [(None, None), ("0", False), ("1", True)])
def test_simd_env_toggle(flag, expected):
    data = run_helper(flag)
    if expected is not None:
        assert data["simd"] is expected
    assert pytest.approx(data["sum"]) == 32.0
    assert pytest.approx(data["mean"]) == pytest.approx(32.0 / 64)
    assert pytest.approx(data["scale_first"]) == 0.0


def test_scalar_broadcast_add_strided():
    env = os.environ.copy()
    env["RAPTORS_SIMD"] = "disable"
    subprocess.check_call([sys.executable, "-c", SCALAR_BROADCAST_CODE], env=env)
