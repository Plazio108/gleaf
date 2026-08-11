"""High-Precision Microsecond Benchmark Script for Gleaf Backends."""

import time
import random
import io
import sys

from gleaf.backends.pure_python import PurePythonCanvas
from gleaf.backends.numpy_backend import NumPyCanvas
from gleaf.backends.numba_backend import NumbaCanvas, warmup_numba_jit

try:
    from gleaf.backends.rich_backend import RichCanvas
except ImportError:
    RichCanvas = None

try:
    from gleaf.backends.curses_backend import CursesCanvas
except ImportError:
    CursesCanvas = None


class DummyStream:
    """Mock stream to isolate compute/formatting performance from terminal I/O bottlenecks."""
    def __init__(self):
        self._buffer = io.BytesIO()

    def write(self, data):
        if isinstance(data, str):
            self._buffer.write(data.encode('utf-8', errors='ignore'))
        elif isinstance(data, (bytes, bytearray)):
            self._buffer.write(data)

    def flush(self):
        pass

    @property
    def buffer(self):
        return self._buffer


def run_benchmark(canvas_cls, name, width=80, height=24, iterations=300):
    if canvas_cls is None:
        print(f"[-] Skipping {name} (Not available)")
        return

    print(f"[*] Initializing {name} ({width}x{height}) ...", end="", flush=True)

    # Warm up JIT compiler for Numba
    if name == "Numba JIT" and warmup_numba_jit:
        warmup_numba_jit()

    try:
        canvas = canvas_cls(width, height)
    except Exception as e:
        print(f" [FAILED: {e}]")
        return

    print(" Done.")

    # Pre-populate initial layout
    canvas.clear()
    canvas.put_str(0, 0, f"Benchmark Test Suite - {name}", fg=(0, 255, 128))

    dummy_out = DummyStream()
    old_stdout = sys.stdout
    old___stdout__ = sys.__stdout__

    try:
        # Initial warm-up render pass (isolated to dummy stream)
        sys.stdout = dummy_out
        sys.__stdout__ = dummy_out
        canvas.render()

        # Timed active rendering loop with changing content (delta simulation)
        start_time = time.perf_counter_ns()

        for i in range(iterations):
            x = (i * 2) % (width - 10)
            y = i % height
            canvas.put_str(x, y, f"F:{i:03d}", fg=(random.randint(60, 255), 120, 220))
            canvas.render()

        end_time = time.perf_counter_ns()

    finally:
        # Restore stdout immediately so results print properly
        sys.stdout = old_stdout
        sys.__stdout__ = old___stdout__

    total_ns = end_time - start_time
    total_us = total_ns / 1_000.0
    avg_us = total_us / iterations
    fps = 1_000_000.0 / avg_us if avg_us > 0 else 0

    print(f"    -> Total Time:   {total_us:,.2f} µs ({total_us / 1_000:,.2f} ms)")
    print(f"    -> Avg per Frame: {avg_us:,.2f} µs")
    print(f"    -> Est. Max FPS:  {fps:,.1f}\n")


def main():
    width, height = 80, 24
    iterations = 300

    print("==================================================")
    print(f"  Gleaf Backend Microsecond Performance Benchmark")
    print(f"  Grid Resolution: {width}x{height} | Iterations: {iterations}")
    print("==================================================\n")

    backends = [
        (PurePythonCanvas, "Pure Python"),
        (NumPyCanvas, "NumPy Vectorized"),
        (NumbaCanvas, "Numba JIT"),
        (RichCanvas, "Rich Optimized"),
        # Curses excluded here as it requires an active live TTY curses window session
    ]

    for cls, label in backends:
        run_benchmark(cls, label, width, height, iterations)


if __name__ == "__main__":
    main()
