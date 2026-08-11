#!/usr/bin/env python3
"""
Gleaf Engine Interactive Performance Demo
Usage: python demo.py [pure|curses|rich|numpy|numba]
"""

import sys
import time
import math
import colorsys

# -----------------------------------------------------------------------------
# Backend Import Strategy
# -----------------------------------------------------------------------------
from gleaf.backends.pure_python import PurePythonCanvas
from gleaf.styles import Style

try:
    from gleaf.backends.curses_backend import CursesCanvas
except ImportError:
    CursesCanvas = None

try:
    from gleaf.backends.rich_backend import RichCanvas
except ImportError:
    RichCanvas = None

try:
    from gleaf.backends.numpy_backend import NumPyCanvas
except ImportError:
    NumPyCanvas = None

try:
    from gleaf.backends.numba_backend import NumbaCanvas
except ImportError:
    NumbaCanvas = None


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Helper to get 0-255 RGB tuples from HSV color space."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


try:
    from gleaf.backends.numba_backend import NumbaCanvas, warmup_numba_jit
except ImportError:
    NumbaCanvas = None
    warmup_numba_jit = None


def init_backend(backend_name: str):
    backend_name = backend_name.lower()
    if backend_name in ("curses", "c") and CursesCanvas:
        return CursesCanvas(), "Curses"
    if backend_name in ("rich", "r") and RichCanvas:
        return RichCanvas(), "Rich"
    if backend_name in ("numpy", "np") and NumPyCanvas:
        return NumPyCanvas(), "NumPy"
    if backend_name in ("numba", "nb") and NumbaCanvas:
        print("Warming up Numba JIT compiler...")
        warmup_numba_jit()  # Silent compilation on 1x1 dummy arrays
        return NumbaCanvas(), "Numba JIT"

    return PurePythonCanvas(), "Pure Python"

def main():
    backend_choice = sys.argv[1] if len(sys.argv) > 1 else "pure"
    canvas, backend_label = init_backend(backend_choice)

    # Alternate screen setup
    canvas.enter_alternate_screen()

    # Bouncing box state
    box_w, box_h = 16, 5
    box_x, box_y = 5.0, 3.0
    vx, vy = 22.0, 12.0  # velocity in cells per second

    # Timing and FPS counter state
    start_time = time.time()
    last_time = start_time
    frame_count = 0
    fps = 0.0
    fps_last_update = start_time

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            t = now - start_time

            # Update FPS every 0.2 seconds
            frame_count += 1
            if now - fps_last_update >= 0.2:
                fps = frame_count / (now - fps_last_update)
                frame_count = 0
                fps_last_update = now

            # Handle Terminal Resize
            canvas.auto_resize()
            w, h = canvas.width, canvas.height

            # -----------------------------------------------------------------
            # 1. Clear Canvas
            # -----------------------------------------------------------------
            canvas.clear()

            # -----------------------------------------------------------------
            # 2. Draw Animated Header & Smooth Horizontal Gradient
            # -----------------------------------------------------------------
            header_str = f" GLEAF ENGINE DEMO | Backend: {backend_label} "
            canvas.put_str(2, 1, header_str, style=Style.BOLD)

            # Smooth animated rainbow gradient bar
            grad_y = 3
            grad_width = min(60, w - 4)
            for col in range(grad_width):
                hue = (col / grad_width + t * 0.3) % 1.0
                fg_color = hsv_to_rgb(hue, 0.9, 1.0)
                bg_color = hsv_to_rgb(hue, 0.8, 0.25)
                canvas.put_str(2 + col, grad_y, "━", fg=fg_color, bg=bg_color, style=Style.BOLD)

            # -----------------------------------------------------------------
            # 3. Update & Draw Bouncing Element
            # -----------------------------------------------------------------
            box_x += vx * dt
            box_y += vy * dt

            # Bounce bounds checking
            max_x = max(0, w - box_w)
            max_y = max(0, h - box_h)

            if box_x <= 0:
                box_x = 0
                vx = abs(vx)
            elif box_x >= max_x:
                box_x = max_x
                vx = -abs(vx)

            if box_y <= 4:  # Keep below header/gradient
                box_y = 4
                vy = abs(vy)
            elif box_y >= max_y:
                box_y = max_y
                vy = -abs(vy)

            bx, by = int(box_x), int(box_y)

            # Dynamic color for bouncing element
            pulse_hue = (t * 0.2) % 1.0
            box_bg = hsv_to_rgb(pulse_hue, 0.7, 0.35)
            box_fg = hsv_to_rgb(pulse_hue + 0.5, 0.8, 1.0)

            # Paint background region for box
            canvas.edit_region_colors(bx, by, box_w, box_h, bg=box_bg)

            # Draw box border & label
            canvas.put_str(bx, by, "+" + "-" * (box_w - 2) + "+", fg=box_fg, style=Style.BOLD)
            for row in range(1, box_h - 1):
                canvas.put_str(bx, by + row, "|", fg=box_fg)
                canvas.put_str(bx + box_w - 1, by + row, "|", fg=box_fg)

            canvas.put_str(bx + 2, by + 1, "BOUNCING", fg=(255, 255, 255), style=Style.BOLD)
            canvas.put_str(bx + 2, by + 2, f"X:{bx:02d} Y:{by:02d}", fg=box_fg)
            canvas.put_str(bx, by + box_h - 1, "+" + "-" * (box_w - 2) + "+", fg=box_fg, style=Style.BOLD)

            # -----------------------------------------------------------------
            # 4. Pulsating Color Grid / Waves
            # -----------------------------------------------------------------
            grid_y = max(by + box_h + 1, 10)
            if grid_y < h - 4:
                canvas.put_str(2, grid_y, "Pulsating Wave Grid:", style=Style.UNDERLINE)
                for gy in range(grid_y + 1, min(grid_y + 4, h - 2)):
                    for gx in range(2, min(40, w - 2)):
                        wave = math.sin(t * 4.0 + gx * 0.3 + gy * 0.5)
                        brightness = int((wave + 1) * 0.5 * 255)
                        canvas.put_str(
                            gx, gy, "#", 
                            fg=(brightness, 120, 255 - brightness),
                            style=Style.DIM if brightness < 100 else Style.NONE
                        )

            # -----------------------------------------------------------------
            # 5. Render FPS Counter & Stats (Top Right)
            # -----------------------------------------------------------------
            fps_color = (0, 255, 120) if fps >= 30 else (255, 200, 0) if fps >= 15 else (255, 50, 50)
            fps_text = f" FPS: {fps:5.1f} | Term: {w}x{h} "
            fps_x = max(0, w - len(fps_text) - 1)

            canvas.edit_region_colors(fps_x, 1, len(fps_text), 1, bg=(20, 30, 45))
            canvas.put_str(fps_x, 1, fps_text, fg=fps_color, style=Style.BOLD)

            # Footer hint
            canvas.put_str(2, h - 1, "Press Ctrl+C to exit", fg=(120, 120, 120))

            # -----------------------------------------------------------------
            # 6. Flush Frame
            # -----------------------------------------------------------------
            canvas.render()

            # Target ~60 FPS update rate limit
            time.sleep(0.016)

    except KeyboardInterrupt:
        pass
    finally:
        canvas.exit_alternate_screen()
        print("\nExited clean.")


if __name__ == "__main__":
    main()
