"""High-Performance NumPy Backend with Vectorized Frame Buffering."""

import sys
import shutil
import numpy as np
from typing import Optional, Tuple
from .base import BaseCanvas, UNSET

try:
    import termios
    import tty
except ImportError:
    termios = None


class NumPyCanvas(BaseCanvas):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        w = width if width is not None else shutil.get_terminal_size().columns
        h = height if height is not None else shutil.get_terminal_size().lines
        super().__init__(w, h)

        self._init_buffers(self.width, self.height)

    def _init_buffers(self, w: int, h: int) -> None:
        """Allocates backend struct-of-arrays memory layout."""
        # Back buffers
        self.b_char = np.full((h, w), 32, dtype=np.uint32)  # Ordinal 32 = ' '
        self.b_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_has_fg = np.zeros((h, w), dtype=np.uint8)

        self.b_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_has_bg = np.zeros((h, w), dtype=np.uint8)

        self.b_style = np.zeros((h, w), dtype=np.uint8)

        # Front buffers (initialized with sentinel values to force initial full render)
        self.f_char = np.full((h, w), 0xFFFFFFFF, dtype=np.uint32)
        self.f_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_has_fg = np.full((h, w), 255, dtype=np.uint8)

        self.f_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_has_bg = np.full((h, w), 255, dtype=np.uint8)

        self.f_style = np.full((h, w), 255, dtype=np.uint8)

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return chr(self.b_char[y, x])
        return " "

    def get_fg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if 0 <= x < self.width and 0 <= y < self.height and self.b_has_fg[y, x]:
            return (int(self.b_fg_r[y, x]), int(self.b_fg_g[y, x]), int(self.b_fg_b[y, x]))
        return None

    def get_bg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if 0 <= x < self.width and 0 <= y < self.height and self.b_has_bg[y, x]:
            return (int(self.b_bg_r[y, x]), int(self.b_bg_g[y, x]), int(self.b_bg_b[y, x]))
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.b_style[y, x])
        return 0

    # --- Vectorized Region / Zone Inspection Overrides ---
    def _clamp_region(self, x: int, y: int, w: int, h: int) -> Tuple[slice, slice]:
        x_start = max(0, x)
        x_end = min(self.width, x + w)
        y_start = max(0, y)
        y_end = min(self.height, y + h)
        return slice(y_start, y_end), slice(x_start, x_end)

    def get_region_chars(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        return np.vectorize(chr, otypes=[str])(self.b_char[s_y, s_x])

    def get_region_fg(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        sub_has = self.b_has_fg[s_y, s_x]
        sub_r, sub_g, sub_b = self.b_fg_r[s_y, s_x], self.b_fg_g[s_y, s_x], self.b_fg_b[s_y, s_x]
        
        res = np.empty(sub_has.shape, dtype=object)
        mask = (sub_has == 1)
        res[~mask] = None
        if np.any(mask):
            rgb_tuples = list(zip(sub_r[mask], sub_g[mask], sub_b[mask]))
            res[mask] = [(int(r), int(g), int(b)) for r, g, b in rgb_tuples]
        return res

    def get_region_bg(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        sub_has = self.b_has_bg[s_y, s_x]
        sub_r, sub_g, sub_b = self.b_bg_r[s_y, s_x], self.b_bg_g[s_y, s_x], self.b_bg_b[s_y, s_x]
        
        res = np.empty(sub_has.shape, dtype=object)
        mask = (sub_has == 1)
        res[~mask] = None
        if np.any(mask):
            rgb_tuples = list(zip(sub_r[mask], sub_g[mask], sub_b[mask]))
            res[mask] = [(int(r), int(g), int(b)) for r, g, b in rgb_tuples]
        return res

    def get_region_styles(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        return self.b_style[s_y, s_x].copy()

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width == self.width and new_height == self.height:
            return
        self.width = new_width
        self.height = new_height
        self._init_buffers(self.width, self.height)

    def auto_resize(self) -> bool:
        term_size = shutil.get_terminal_size()
        if term_size.columns != self.width or term_size.lines != self.height:
            self.resize(term_size.columns, term_size.lines)
            return True
        return False

    def enter_alternate_screen(self) -> None:
        sys.stdout.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.stdout.flush()

        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def exit_alternate_screen(self) -> None:
        sys.stdout.write("\033[0m\033[?1049l\033[?25h")
        sys.stdout.flush()

        if termios is not None and hasattr(self, "_old_term") and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def clear(self) -> None:
        self.b_char.fill(32)
        self.b_has_fg.fill(0)
        self.b_has_bg.fill(0)
        self.b_style.fill(0)

    def put_str(self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET) -> None:
        if y < 0 or y >= self.height or not text:
            return

        x_start = max(0, x)
        x_end = min(self.width, x + len(text))
        if x_start >= x_end:
            return

        text_offset_start = x_start - x
        text_offset_end = text_offset_start + (x_end - x_start)
        target_text = text[text_offset_start:text_offset_end]

        # Vectorized Unicode conversion to uint32
        char_ords = np.fromiter((ord(c) for c in target_text), dtype=np.uint32, count=len(target_text))
        self.b_char[y, x_start:x_end] = char_ords

        if fg is not UNSET:
            if fg is None:
                self.b_has_fg[y, x_start:x_end] = 0
            else:
                self.b_has_fg[y, x_start:x_end] = 1
                self.b_fg_r[y, x_start:x_end] = fg[0]
                self.b_fg_g[y, x_start:x_end] = fg[1]
                self.b_fg_b[y, x_start:x_end] = fg[2]

        if bg is not UNSET:
            if bg is None:
                self.b_has_bg[y, x_start:x_end] = 0
            else:
                self.b_has_bg[y, x_start:x_end] = 1
                self.b_bg_r[y, x_start:x_end] = bg[0]
                self.b_bg_g[y, x_start:x_end] = bg[1]
                self.b_bg_b[y, x_start:x_end] = bg[2]

        if style is not UNSET:
            self.b_style[y, x_start:x_end] = 0 if style is None else style

    def edit_region_colors(self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, style=UNSET) -> None:
        x_start = max(0, x)
        x_end = min(self.width, x + w)
        y_start = max(0, y)
        y_end = min(self.height, y + h)

        if x_start >= x_end or y_start >= y_end:
            return

        region = (slice(y_start, y_end), slice(x_start, x_end))

        if fg is not UNSET:
            if fg is None:
                self.b_has_fg[region] = 0
            else:
                self.b_has_fg[region] = 1
                self.b_fg_r[region] = fg[0]
                self.b_fg_g[region] = fg[1]
                self.b_fg_b[region] = fg[2]

        if bg is not UNSET:
            if bg is None:
                self.b_has_bg[region] = 0
            else:
                self.b_has_bg[region] = 1
                self.b_bg_r[region] = bg[0]
                self.b_bg_g[region] = bg[1]
                self.b_bg_b[region] = bg[2]

        if style is not UNSET:
            self.b_style[region] = 0 if style is None else style

    def _style_to_sgr(self, flags: int) -> str:
        sgr = []
        if flags & 1:  sgr.append("1")  # Bold
        if flags & 2:  sgr.append("2")  # Dim
        if flags & 4:  sgr.append("3")  # Italic
        if flags & 8:  sgr.append("4")  # Underline
        if flags & 16: sgr.append("5")  # Blink
        if flags & 32: sgr.append("7")  # Reverse
        return ";".join(sgr)

    def render(self) -> None:
        diff = (
            (self.b_char != self.f_char) |
            (self.b_has_fg != self.f_has_fg) |
            (self.b_has_bg != self.f_has_bg) |
            (self.b_style != self.f_style) |
            ((self.b_has_fg == 1) & ((self.b_fg_r != self.f_fg_r) | (self.b_fg_g != self.f_fg_g) | (self.b_fg_b != self.f_fg_b))) |
            ((self.b_has_bg == 1) & ((self.b_bg_r != self.f_bg_r) | (self.b_bg_g != self.f_bg_g) | (self.b_bg_b != self.f_bg_b)))
        )

        if not np.any(diff):
            return

        y_indices, x_indices = np.where(diff)
        buf = []
        app = buf.append

        last_y, last_x = -1, -1
        for y, x in zip(y_indices, x_indices):
            if y != last_y or x != last_x + 1:
                app(f"\033[{y+1};{x+1}H")

            sgr_codes = ["0"]

            st = self.b_style[y, x]
            if st > 0:
                sgr = self._style_to_sgr(st)
                if sgr:
                    sgr_codes.append(sgr)

            if self.b_has_fg[y, x]:
                sgr_codes.append(f"38;2;{self.b_fg_r[y, x]};{self.b_fg_g[y, x]};{self.b_fg_b[y, x]}")

            if self.b_has_bg[y, x]:
                sgr_codes.append(f"48;2;{self.b_bg_r[y, x]};{self.b_bg_g[y, x]};{self.b_bg_b[y, x]}")

            app(f"\033[{';'.join(sgr_codes)}m{chr(self.b_char[y, x])}")

            last_y, last_x = y, x

        # Synchronize front buffer
        self.f_char[diff] = self.b_char[diff]
        self.f_has_fg[diff] = self.b_has_fg[diff]
        self.f_fg_r[diff] = self.b_fg_r[diff]
        self.f_fg_g[diff] = self.b_fg_g[diff]
        self.f_fg_b[diff] = self.b_fg_b[diff]
        self.f_has_bg[diff] = self.b_has_bg[diff]
        self.f_bg_r[diff] = self.b_bg_r[diff]
        self.f_bg_g[diff] = self.b_bg_g[diff]
        self.f_bg_b[diff] = self.b_bg_b[diff]
        self.f_style[diff] = self.b_style[diff]

        if buf:
            sys.stdout.write("".join(buf))
            sys.stdout.flush()
