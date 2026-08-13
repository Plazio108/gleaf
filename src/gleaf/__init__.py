"""
gleaf - Adaptive High-Performance Terminal Canvas.
Supports numba, numpy, curses, rich, and pure python fallbacks.
"""

from .backends.pure_python import PurePythonCanvas
from .styles import Style, Modifiers
from .caps import TerminalCaps
from .utils import managed_canvas, enable_suspend, handoff

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
        # if HAS_NUMBA:
        #     return NumbaCanvas, "Numba (Parallel JIT)"
        if HAS_NUMPY:
            return NumPyCanvas, "NumPy (Vectorized)"
        if HAS_RICH:
            return RichCanvas, "Rich (Segment Builder)"
        if HAS_CURSES:
            return CursesCanvas, "Curses (Native)"
        return PurePythonCanvas, "Pure Python (Zero-Dependency)"

    # Manual forces
    # if backend == "numba" and HAS_NUMBA:
    #     return NumbaCanvas, "Numba"
    if backend == "numpy" and HAS_NUMPY:
        return NumPyCanvas, "NumPy"
    # if backend == "fast_numpy" and HAS_NUMPY:
    #     return FastNumPyCanvas, "Fast NumPy"
    if backend == "rich" and HAS_RICH:
        return RichCanvas, "Rich"
    if backend == "curses" and HAS_CURSES:
        return CursesCanvas, "Curses"
    if backend in ("pure", "python"):
        return PurePythonCanvas, "Pure Python"

    raise ImportError(
        f"Backend '{backend}' requested but dependencies are missing.")


def TerminalCanvas(width: int = None, height: int = None, backend: str = "auto"):
    cls, _ = get_canvas_class(backend)
    return cls(width=width, height=height)


__all__ = [
    "TerminalCanvas", "get_canvas_class", "Style", "Modifiers", "TerminalCaps",
    "managed_canvas", "enable_suspend", "handoff"
]
