#!/usr/bin/env python3
"""
Gleaf Engine - Terminal State & Handoff Diagnostic Tool
Usage: python test_utils.py [pure|curses|rich]
"""

import sys
import time
import subprocess

from gleaf.caps import Modifiers
from gleaf.utils import managed_canvas, handoff, enable_suspend

from gleaf.backends.pure_python import PurePythonCanvas
try:
    from gleaf.backends.curses_backend import CursesCanvas
except ImportError:
    CursesCanvas = None
try:
    from gleaf.backends.rich_backend import RichCanvas
except ImportError:
    RichCanvas = None


def init_backend(backend_name: str):
    backend_name = backend_name.lower()
    if backend_name in ("curses", "c") and CursesCanvas:
        return CursesCanvas(), "Curses"
    if backend_name in ("rich", "r") and RichCanvas:
        return RichCanvas(), "Rich"
    return PurePythonCanvas(), "Pure Python"


def main():
    backend_choice = sys.argv[1] if len(sys.argv) > 1 else "pure"
    canvas, backend_label = init_backend(backend_choice)

    # Enable SIGTSTP (Ctrl+Z) graceful suspension handler
    enable_suspend(canvas, on_resume=lambda: print("[Resumed from background]"))

    # Wrap main execution loop in managed_canvas for crash safety
    with managed_canvas(canvas):
        start_time = time.time()
        message = "Press 'H' to test Handoff | 'Ctrl+Z' to suspend | 'Ctrl+C' to exit"
        
        running = True
        while running:
            t = time.time() - start_time
            if hasattr(canvas, "auto_resize"):
                canvas.auto_resize()
            w, h = canvas.width, canvas.height

            canvas.clear()

            # Render UI Frame
            canvas.put_str(2, 1, f" GLEAF UTILS DIAGNOSTIC | Backend: {backend_label} ", style=Modifiers.BOLD)
            canvas.put_str(2, 3, message, fg=(200, 200, 100))

            # Pulsating indicator box
            box_y = 6
            char_glow = "O" if int(t * 4) % 2 == 0 else "o"
            canvas.put_str(2, box_y, f"[{char_glow}] Main loop active. Uptime: {t:5.1f}s", fg=(100, 255, 100))

            canvas.put_str(2, h - 1, "Controls: [H]andoff to shell | [Ctrl+C] Quit", fg=(120, 120, 120))
            canvas.render()

            # Simple non-blocking or low-latency input check simulation
            # For testing handoff interactively via a keypress or timed trigger:
            if int(t) > 0 and int(t) % 10 == 0:
                # Trigger an automated demonstration handoff every 10 seconds
                canvas.put_str(2, 5, "Triggering automatic temporary terminal handoff...", fg=(255, 100, 100))
                canvas.render()
                time.sleep(1.0)

                # --- Handoff Demonstration ---
                with handoff(canvas):
                    print("\n--- [START] TEMPORARY TERMINAL HANDOFF ---")
                    print("The TUI is suspended. You have raw terminal control.")
                    print("Executing external process (e.g., listing files or running a command)...")
                    subprocess.run(["uname", "-a"])
                    print("--- [END] RESTORING TUI CANVAS IN 3 SECONDS ---\n")
                    time.sleep(3.0)
                # -----------------------------
                
                start_time = time.time()  # reset timer after handoff

            time.sleep(0.033)


if __name__ == "__main__":
    main()
