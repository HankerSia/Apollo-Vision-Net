"""Lightweight MapTR components.

This repo only needs a small subset of MapTR modules (e.g. MapTRDecoder)
for config compatibility. Heavy CUDA ops and unrelated modules are
intentionally not vendored here.
"""

from .modules import *  # noqa: F401,F403
