#!/usr/bin/env python3
"""
Gleaf Engine - Modifiers & Capability Diagnostic Tool
Usage: python diag_modifiers.py [pure|curses|rich|numpy|numba]
"""

import sys
import time
import colorsys
from typing import Tuple

# -----------------------------------------------------------------------------
# Backend Import Strategy
# -----------------------------------------------------------------------------
from gleaf.backends.pure_python import PurePythonCanvas
from gleaf.caps import Modifiers, TerminalCaps

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
    from gleaf.backends.fast_numpy_backend import FastNumPyCanvas
except ImportError:
    NumPyCanvas = None
    FastNumPyCanvas = None

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
    if backend_name in ("fast_numpy", "fnp") and FastNumPyCanvas:
        return FastNumPyCanvas(), "Fast NumPy"
    if backend_name in ("numba", "nb") and NumbaCanvas:
        if warmup_numba_jit:
            warmup_numba_jit()
        return NumbaCanvas(), "Numba JIT"

    return PurePythonCanvas(), "Pure Python"


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def main():
    backend_choice = sys.argv[1] if len(sys.argv) > 1 else "pure"
    canvas, backend_label = init_backend(backend_choice)
    caps = getattr(canvas, "caps", TerminalCaps())

    # Modifiers test suite table definition
    # (Label, Bitmask Flag, Test Sample Text)
    test_cases = [
        # Standard Modifiers
        ("BOLD", Modifiers.BOLD, "The quick brown fox"),
        ("DIM", Modifiers.DIM, "The quick brown fox"),
        ("ITALIC", Modifiers.ITALIC, "The quick brown fox"),
        ("UNDERLINE", Modifiers.UNDERLINE, "The quick brown fox"),
        ("REVERSE", Modifiers.REVERSE, " The quick brown fox "),
        ("STANDOUT", Modifiers.STANDOUT, " The quick brown fox "),
        ("BLINK", Modifiers.BLINK, "The quick brown fox"),
        ("STRIKETHROUGH", Modifiers.STRIKETHROUGH, "The quick brown fox"),
        ("OVERLINE", Modifiers.OVERLINE, "The quick brown fox"),
        ("HIDDEN", Modifiers.HIDDEN, "[HIDDEN TEXT HERE]"),
        
        # Extended Underlines
        ("DOUBLE_UNDERLINE", Modifiers.DOUBLE_UNDERLINE, "The quick brown fox"),
        ("CURLY_UNDERLINE", Modifiers.CURLY_UNDERLINE, "Wavy / Curly Error Line"),
        ("DOTTED_UNDERLINE", Modifiers.DOTTED_UNDERLINE, "Dotted Underline Text"),
        ("DASHED_UNDERLINE", Modifiers.DASHED_UNDERLINE, "Dashed Underline Text"),
        
        # Combinations
        ("BOLD + ITALIC", Modifiers.BOLD | Modifiers.ITALIC, "Bold Italic Combo"),
        ("BOLD + REVERSE", Modifiers.BOLD | Modifiers.REVERSE, " Bold Reversed Combo "),
        ("ITALIC + CURLY", Modifiers.ITALIC | Modifiers.CURLY_UNDERLINE, "Italic Curly Combo"),
        ("REVERSE + CURLY", Modifiers.REVERSE | Modifiers.CURLY_UNDERLINE, " Reverse Curly Combo "),
    ]

    canvas.enter_alternate_screen()
    start_time = time.time()

    try:
        while True:
            t = time.time() - start_time

            # Handle terminal resize
            if hasattr(canvas, "auto_resize"):
                canvas.auto_resize()
            w, h = canvas.width, canvas.height

            canvas.clear()

            # -----------------------------------------------------------------
            # 1. Header & Terminal Capability Badges
            # -----------------------------------------------------------------
            title = f" GLEAF MODIFIERS DIAGNOSTIC | Backend: {backend_label} "
            canvas.put_str(2, 1, title, style=Modifiers.BOLD)

            # Capability Badges
            tc_status = "TRUECOLOR" if caps.has_truecolor else "256-COLOR" if caps.has_256color else "NO-COLOR"
            ext_ul_status = "EXT-UNDERLINE: YES" if caps.has_extended_underline else "EXT-UNDERLINE: NO (Fallback)"
            badge_str = f" [{tc_status}] [{ext_ul_status}] "
            canvas.put_str(max(2, w - len(badge_str) - 2), 1, badge_str, fg=(120, 220, 255), style=Modifiers.DIM)

            # Table Header
            canvas.put_str(2, 3, "FLAG NAME", fg=(180, 180, 180), style=Modifiers.BOLD | Modifiers.UNDERLINE)
            canvas.put_str(22, 3, "HEX BIT", fg=(180, 180, 180), style=Modifiers.BOLD | Modifiers.UNDERLINE)
            canvas.put_str(34, 3, "RENDERED SAMPLE OUTPUT", fg=(180, 180, 180), style=Modifiers.BOLD | Modifiers.UNDERLINE)

            # -----------------------------------------------------------------
            # 2. Render Modifier Rows
            # -----------------------------------------------------------------
            row_y = 5
            for label, flag, sample in test_cases:
                if row_y >= h - 4:
                    break  # Prevent viewport overflow

                # Highlight bitmask hex code
                hex_str = f"0x{int(flag):08X}"
                
                canvas.put_str(2, row_y, label, fg=(220, 220, 100))
                canvas.put_str(22, row_y, hex_str, fg=(120, 120, 120), style=Modifiers.DIM)

                # Pulsating color for wavy/curly spellcheck underlines
                ul_color = hsv_to_rgb(t * 0.5, 0.9, 1.0) if "CURLY" in label else None

                # Render sample with target modifier
                try:
                    canvas.put_str(
                        34, row_y, sample,
                        fg=(255, 255, 255),
                        style=flag,
                        ul_fg=ul_color
                    )
                except TypeError:
                    # Fallback for backends without explicit ul_fg parameter
                    canvas.put_str(34, row_y, sample, fg=(255, 255, 255), style=flag)

                row_y += 1

            # -----------------------------------------------------------------
            # 3. Dynamic Underline Color (SGR 58) Test Box
            # -----------------------------------------------------------------
            if row_y + 2 < h - 1:
                row_y += 1
                canvas.put_str(2, row_y, "SGR 58 Underline Color Test:", style=Modifiers.BOLD)
                row_y += 1
                
                rainbow_sample = "Rainbow Spectrum Wavy Underline Test Sequence"
                for i, char in enumerate(rainbow_sample):
                    hue = (t * 0.3 + i * 0.05) % 1.0
                    char_ul_fg = hsv_to_rgb(hue, 1.0, 1.0)
                    try:
                        canvas.put_str(
                            2 + i, row_y, char,
                            fg=(220, 220, 220),
                            style=Modifiers.CURLY_UNDERLINE,
                            ul_fg=char_ul_fg
                        )
                    except TypeError:
                        canvas.put_str(2 + i, row_y, char, style=Modifiers.CURLY_UNDERLINE)

            # Footer
            canvas.put_str(2, h - 1, "Press Ctrl+C to exit", fg=(100, 100, 100))

            # -----------------------------------------------------------------
            # 4. Flush Frame
            # -----------------------------------------------------------------
            canvas.render()
            time.sleep(0.033)  # ~30 FPS refresh

    except KeyboardInterrupt:
        pass
    finally:
        canvas.exit_alternate_screen()
        print("Diagnostic exited cleanly.")


if __name__ == "__main__":
    main()
