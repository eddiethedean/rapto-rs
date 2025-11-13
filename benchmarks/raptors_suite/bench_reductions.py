"""
ASV benchmarks for global and axis reductions.
"""

from __future__ import annotations

import numpy as np
import raptors

from .utils import configure_env, make_inputs


class ReductionSuite:
    params = [
        [(512, 512), (1024, 1024)],
        ["float32", "float64"],
        ["contiguous", "transpose"],
        ["sum", "mean", "mean_axis0", "mean_axis1"],
        ["auto", "force"],
        ["auto", 1, 4],
    ]
    param_names = ["shape", "dtype", "layout", "operation", "simd_mode", "threads"]

    def setup(self, shape, dtype, layout, operation, simd_mode, threads):
        configure_env(threads=threads, simd_mode=simd_mode)
        self.numpy_array, self.raptors_array = make_inputs(shape, dtype, layout)
        self.operation = operation

    def _numpy_op(self):
        op = self.operation
        if op == "sum":
            return float(self.numpy_array.sum())
        if op == "mean":
            return float(self.numpy_array.mean())
        if op == "mean_axis0":
            return np.asarray(self.numpy_array.mean(axis=0))
        if op == "mean_axis1":
            return np.asarray(self.numpy_array.mean(axis=1))
        raise ValueError(f"Unknown operation {op}")

    def _raptors_op(self):
        op = self.operation
        if op == "sum":
            return self.raptors_array.sum()
        if op == "mean":
            return self.raptors_array.mean()
        if op == "mean_axis0":
            return self.raptors_array.mean_axis(0)
        if op == "mean_axis1":
            return self.raptors_array.mean_axis(1)
        raise ValueError(f"Unknown operation {op}")

    def time_numpy(self, *params):
        self._numpy_op()

    def time_raptors(self, *params):
        self._raptors_op()

