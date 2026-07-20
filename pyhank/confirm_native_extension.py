try:
    from pyhank import _pyhank_native
except ImportError:
    raise RuntimeError("Native extension not available")

if not _pyhank_native.is_release_build():  # type: ignore
    raise RuntimeError("Native extension available, but not built in release mode")
