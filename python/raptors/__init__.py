"""
Python interface for the Rust-powered Raptors array library.

This placeholder module wires through to the Rust extension module generated
by `maturin`, exposing an initial `array` constructor stub.
"""

from __future__ import annotations

from . import _raptors as _core

__all__ = ["array", "__version__", "__author__", "__github__"]

__version__ = getattr(_core, "__version__", "0.0.1")
__author__ = getattr(_core, "__author__", "Odos Matthews <odosmatthews@gmail.com>")
__github__ = getattr(_core, "__github__", "https://github.com/eddiethedean")


def array() -> None:
    """Create a new Raptors array (placeholder implementation)."""
    return _core.rustarray_new()

