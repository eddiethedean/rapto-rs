"""
ASV benchmarks for scale operations.
"""

from __future__ import annotations

from .utils import configure_env, make_inputs, scale_factor


class ScaleSuite:
    params = [
        [(512, 512), (1024, 1024), (2048, 2048)],
        ["float32", "float64"],
        ["contiguous", "transpose"],
        ["auto", "force", "disable"],
        ["auto", 1, 4],
    ]
    param_names = ["shape", "dtype", "layout", "simd_mode", "threads"]

    def setup(self, shape, dtype, layout, simd_mode, threads):
        configure_env(threads=threads, simd_mode=simd_mode)
        self.numpy_array, self.raptors_array = make_inputs(shape, dtype, layout)
        self.factor = scale_factor(dtype)

    def time_numpy_scale(self, *params):
        # NumPy broadcast multiply
        _ = self.numpy_array * self.factor

    def time_raptors_scale(self, *params):
        self.raptors_array.scale(self.factor)

