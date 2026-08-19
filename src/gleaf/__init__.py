"""
gleaf - Adaptive High-Performance Terminal Canvas.
Supports numba, numpy, curses, rich, and pure python fallbacks.
"""

from .backends.base import BaseCanvas, BaseTexture, UNSET
from .backends.pure_python import PurePythonCanvas
from .caps import TerminalCaps
from .styles import Modifiers, Style
from .utils import enable_suspend, handoff, managed_canvas

HAS_NUMBA = HAS_NUMPY = HAS_CURSES = HAS_RICH = False

try:
    from .backends.numpy_backend import NumPyCanvas

    HAS_NUMPY = True
except ImportError:
    pass

try:
    from .backends.rich_backend import RichCanvas

    HAS_RICH = True
except ImportError:
    pass

try:
    from .backends.curses_backend import CursesCanvas

    HAS_CURSES = True
except ImportError:
    pass


def get_canvas_class(backend: str = "auto"):
    """Resolves and returns the canvas class based on selection."""
    backend = backend.lower().strip()
    if backend == "auto":
        if HAS_NUMPY:
            return NumPyCanvas, "NumPy (Vectorized)"
        return PurePythonCanvas, "Pure Python (Zero-Dependency)"

    if backend == "numpy" and HAS_NUMPY:
        return NumPyCanvas, "NumPy"
    if backend == "rich" and HAS_RICH:
        return RichCanvas, "Rich"
    if backend == "curses" and HAS_CURSES:
        return CursesCanvas, "Curses"
    if backend in ("pure", "python"):
        return PurePythonCanvas, "Pure Python"

    raise ImportError(f"Backend '{backend}' requested but dependencies are missing.")


def get_texture_class(backend: str = "auto"):
    """Resolves the corresponding Texture class for the active backend."""
    cls, _ = get_canvas_class(backend)
    if cls.__name__ == "NumPyCanvas":
        from .backends.numpy_backend import NumpyTexture

        return NumpyTexture
    else:
        from .backends.pure_python import PurePythonTexture

        return PurePythonTexture


def TerminalCanvas(width: int = None, height: int = None, backend: str = "auto"):
    cls, _ = get_canvas_class(backend)
    return cls(width=width, height=height)


def TerminalTexture(
    width: int = None,
    height: int = None,
    backend: str = "auto",
    data_buffer=None,
    matrix=None,
):
    """
    Creates a texture matching the selected backend.
    Can be loaded from mmap/bytes (data_buffer) or a 2D array (matrix).
    """
    cls = get_texture_class(backend)

    if matrix is not None:
        return cls.from_matrix(matrix)

    # Needs valid dimensions if creating empty or from bytes
    if width is None or height is None:
        raise ValueError("Texture requires explicit width and height.")

    if data_buffer is not None:
        return cls(width, height, data_buffer=data_buffer)
    return cls(width, height)


__all__ = [
    "UNSET",
    "BaseCanvas",
    "BaseTexture",
    "Modifiers",
    "Style",
    "TerminalCanvas",
    "TerminalCaps",
    "TerminalTexture",
    "enable_suspend",
    "get_canvas_class",
    "handoff",
    "managed_canvas",
]
