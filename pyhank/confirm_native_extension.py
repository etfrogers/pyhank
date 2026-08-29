try:
    from pyhank import _pyhank_native
except ImportError as err:
    raise RuntimeError("Native extension not available") from err

if not _pyhank_native.is_release_build():  # type: ignore
    raise RuntimeError("Native extension available, but not built in release mode")
