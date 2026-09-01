from importlib.metadata import PackageNotFoundError, version

try:
    from ._pyhank_native import HankelTransform, iqdht, qdht
except ImportError:  # pragma: no cover
    from ._pure_python import HankelTransform, iqdht, qdht  # pragma: no cover

try:
    __version__ = version("pyhank")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

__all__ = ["HankelTransform", "__version__", "iqdht", "qdht"]
