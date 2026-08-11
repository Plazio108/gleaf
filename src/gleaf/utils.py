"""Terminal state management, error restoration, and handoff utilities."""

import os
import sys
import signal
from contextlib import contextmanager
from typing import Callable, Any


@contextmanager
def managed_canvas(canvas):
    """Context manager ensuring safe terminal recovery on unexpected crashes."""
    canvas.enter_alternate_screen()
    try:
        yield canvas
    except Exception as e:
        canvas.exit_alternate_screen()
        print(f"\n[gleaf] Fatal error caught. Terminal safely restored.\n{e}")
        raise
    finally:
        canvas.exit_alternate_screen()


@contextmanager
def handoff(canvas):
    """
    Context manager that temporarily releases terminal control to external commands,
    then automatically restores the alternate screen and canvas state upon exit.
    """
    canvas.exit_alternate_screen()
    try:
        yield
    finally:
        canvas.enter_alternate_screen()
        canvas.clear()
        canvas.render()


def enable_suspend(canvas, on_resume: Callable = None):
    """Hooks SIGTSTP (Ctrl+Z) to gracefully suspend and resume the canvas."""
    def _suspend_handler(signum, frame):
        canvas.exit_alternate_screen()
        os.kill(os.getpid(), signal.SIGSTOP)
        
        # Resumed via `fg`
        canvas.enter_alternate_screen()
        canvas.clear()
        if on_resume:
            on_resume()
        canvas.render()

    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _suspend_handler)
