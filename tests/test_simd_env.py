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
