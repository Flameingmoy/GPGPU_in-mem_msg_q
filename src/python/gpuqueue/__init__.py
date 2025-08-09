"""
GPUQueue: CUDA-backed GPU-resident message queue (scaffold).

This is a minimal package skeleton; the compiled extension _core provides
`init()`, `shutdown()`, and `version()`.
"""
from ._version import __version__

try:
    from ._core import init, shutdown, version as core_version
except Exception as e:  # pragma: no cover - during source-only operations
    core_version = None
    def init(*_, **__):
        raise RuntimeError("gpuqueue extension not built. Build/install the package.") from e
    def shutdown():
        return None

__all__ = ["__version__", "init", "shutdown", "core_version"]
