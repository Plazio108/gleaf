"""NumPy Struct-of-Arrays (SoA) Double-Buffered Delta Canvas."""

import sys
import numpy as np
from .base import BaseCanvas

try:
    import termios
    import tty
except ImportError:
    termios = None


class NumPyCanvas(BaseCanvas):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self._allocate_soa_buffers()

    def _allocate_soa_buffers(self):
        w, h = self.width, self.height

        # Back Buffers
        self.b_char = np.full((h, w), ord(' '), dtype=np.uint32)
        self.b_has_fg = np.zeros((h, w), dtype=np.bool_)
        self.b_has_bg = np.zeros((h, w), dtype=np.bool_)
        self.b_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_b = np.zeros((h, w), dtype=np.uint8)

        # Front Buffers
        self.f_char = np.zeros((h, w), dtype=np.uint32)
        self.f_has_fg = np.zeros((h, w), dtype=np.bool_)
        self.f_has_bg = np.zeros((h, w), dtype=np.bool_)
        self.f_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_b = np.zeros((h, w), dtype=np.uint8)

    def clear(self):
        if hasattr(self, 'b_char'):
            self.b_char.fill(ord(' '))
            self.b_has_fg.fill(False)
            self.b_has_bg.fill(False)

    def invalidate_front_buffer(self):
        if hasattr(self, 'f_char'):
            self.f_char.fill(0)

    def resize(self, width: int, height: int):
        super().resize(width, height)
        self._allocate_soa_buffers()
        sys.__stdout__.write("\033[2J")
        sys.__stdout__.flush()
        self.invalidate_front_buffer()

    def enter_alternate_screen(self):
        # \033[?1049h = alt screen, \033[H = home, \033[2J = clear, \033[?25l = hide cursor
        sys.__stdout__.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.__stdout__.flush()

        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

        self.invalidate_front_buffer()

    def exit_alternate_screen(self):
        # \033[0m = reset, \033[?1049l = exit alt screen, \033[?25h = show cursor
        sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
        sys.__stdout__.flush()

        if termios is not None and hasattr(self, '_old_term'): # and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def put_str(self, x, y, text, fg=None, bg=None, style=0):
        if y < 0 or y >= self.height:
            return
        length = min(len(text), self.width - x)
        if length <= 0:
            return

        self.b_char[y, x:x+length] = [ord(c) for c in text[:length]]

        if fg is not None:
            self.b_has_fg[y, x:x+length] = True
            self.b_fg_r[y, x:x+length] = fg[0]
            self.b_fg_g[y, x:x+length] = fg[1]
            self.b_fg_b[y, x:x+length] = fg[2]

        if bg is not None:
            self.b_has_bg[y, x:x+length] = True
            self.b_bg_r[y, x:x+length] = bg[0]
            self.b_bg_g[y, x:x+length] = bg[1]
            self.b_bg_b[y, x:x+length] = bg[2]

    def edit_region_colors(self, x, y, w, h, fg=None, bg=None):
        sy = max(0, y); ey = min(self.height, y + h)
        sx = max(0, x); ex = min(self.width, x + w)
        if fg is not None:
            self.b_has_fg[sy:ey, sx:ex] = True
            self.b_fg_r[sy:ey, sx:ex] = fg[0]
            self.b_fg_g[sy:ey, sx:ex] = fg[1]
            self.b_fg_b[sy:ey, sx:ex] = fg[2]
        if bg is not None:
            self.b_has_bg[sy:ey, sx:ex] = True
            self.b_bg_r[sy:ey, sx:ex] = bg[0]
            self.b_bg_g[sy:ey, sx:ex] = bg[1]
            self.b_bg_b[sy:ey, sx:ex] = bg[2]

    def render(self):
        # Compute delta mask using vector operations across all channels
        char_diff = self.b_char != self.f_char
        fg_diff = (self.b_has_fg != self.f_has_fg) | (self.b_has_fg & (
            (self.b_fg_r != self.f_fg_r) | (self.b_fg_g != self.f_fg_g) | (self.b_fg_b != self.f_fg_b)
        ))
        bg_diff = (self.b_has_bg != self.f_has_bg) | (self.b_has_bg & (
            (self.b_bg_r != self.f_bg_r) | (self.b_bg_g != self.f_bg_g) | (self.b_bg_b != self.f_bg_b)
        ))

        diff_mask = char_diff | fg_diff | bg_diff
        if not np.any(diff_mask):
            return

        # Update front buffer where diffs occurred
        self.f_char[diff_mask] = self.b_char[diff_mask]
        self.f_has_fg[diff_mask] = self.b_has_fg[diff_mask]
        self.f_has_bg[diff_mask] = self.b_has_bg[diff_mask]
        self.f_fg_r[diff_mask] = self.b_fg_r[diff_mask]
        self.f_fg_g[diff_mask] = self.b_fg_g[diff_mask]
        self.f_fg_b[diff_mask] = self.b_fg_b[diff_mask]
        self.f_bg_r[diff_mask] = self.b_bg_r[diff_mask]
        self.f_bg_g[diff_mask] = self.b_bg_g[diff_mask]
        self.f_bg_b[diff_mask] = self.b_bg_b[diff_mask]

        out = []
        app = out.append

        row_indices = np.where(np.any(diff_mask, axis=1))[0]

        cur_has_fg = False
        cur_has_bg = False
        cur_fg = (-1, -1, -1)
        cur_bg = (-1, -1, -1)
        cx, cy = -1, -1

        for y in row_indices:
            col_indices = np.where(diff_mask[y])[0]
            for x in col_indices:
                if cx != x or cy != y:
                    app(f"\033[{y+1};{x+1}H")

                has_fg = self.b_has_fg[y, x]
                fg_rgb = (self.b_fg_r[y, x], self.b_fg_g[y, x], self.b_fg_b[y, x])
                if has_fg and (not cur_has_fg or cur_fg != fg_rgb):
                    app(f"\033[38;2;{fg_rgb[0]};{fg_rgb[1]};{fg_rgb[2]}m")
                    cur_has_fg = True
                    cur_fg = fg_rgb
                elif not has_fg and cur_has_fg:
                    app("\033[39m")
                    cur_has_fg = False

                has_bg = self.b_has_bg[y, x]
                bg_rgb = (self.b_bg_r[y, x], self.b_bg_g[y, x], self.b_bg_b[y, x])
                if has_bg and (not cur_has_bg or cur_bg != bg_rgb):
                    app(f"\033[48;2;{bg_rgb[0]};{bg_rgb[1]};{bg_rgb[2]}m")
                    cur_has_bg = True
                    cur_bg = bg_rgb
                elif not has_bg and cur_has_bg:
                    app("\033[49m")
                    cur_has_bg = False

                ch = chr(self.b_char[y, x])
                app(ch)
                cx = x + 1
                cy = y

        if cur_has_fg or cur_has_bg:
            app("\033[0m")

        if out:
            sys.__stdout__.write("".join(out))
            sys.__stdout__.flush()
