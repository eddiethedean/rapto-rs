"""
ASV benchmarks for broadcast additions.
"""

from __future__ import annotations

import numpy as np
import raptors

from .utils import configure_env, make_inputs


class BroadcastSuite:
    params = [
        [(512, 512), (1024, 1024)],
        ["float32", "float64"],
        ["same", "row", "column"],
        ["contiguous", "transpose"],
        ["auto", "force"],
        ["auto", 1, 4],
    ]
    param_names = ["shape", "dtype", "operation", "layout", "simd_mode", "threads"]

    def setup(self, shape, dtype, operation, layout, simd_mode, threads):
        configure_env(threads=threads, simd_mode=simd_mode)
        self.numpy_array, self.raptors_array = make_inputs(shape, dtype, layout)
        self.operation = operation

        if operation == "same":
            rhs_np = np.flip(self.numpy_array, axis=-1).copy()
        elif operation == "row":
            rhs_np = np.arange(self.numpy_array.shape[1], dtype=self.numpy_array.dtype)
        elif operation == "column":
            rhs_np = np.arange(self.numpy_array.shape[0], dtype=self.numpy_array.dtype).reshape(
                -1, 1
            )
        else:
            raise ValueError(f"Unsupported operation {operation}")

        self.rhs_numpy = rhs_np
        self.rhs_raptors = (
            self.raptors_array
            if operation == "same"
            else (raptors.from_numpy(rhs_np) if operation in {"row", "column"} else None)
        )

        if operation == "same":
            self.rhs_raptors = raptors.from_numpy(self.rhs_numpy)

    def time_numpy(self, *params):
        np.add(self.numpy_array, self.rhs_numpy)

    def time_raptors(self, *params):
        raptors.broadcast_add(self.raptors_array, self.rhs_raptors)

