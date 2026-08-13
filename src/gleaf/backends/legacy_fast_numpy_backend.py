"""High-Performance NumPy Backend with Vectorized Cell/Zone Lookups."""

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

# Structured dtype packing a full cell into contiguous memory
# 128-bits total: Char (32b), FG (32b), BG (32b), Style (8b), padding
CELL_DTYPE = np.dtype([
    ('char', np.uint32),
    ('fg', np.uint32),    # Highest bit (0x80000000) = has_color flag, lower 24 = RGB
    ('bg', np.uint32),    # Highest bit (0x80000000) = has_color flag, lower 24 = RGB
    ('style', np.uint8),
])

COLOR_FLAG = 0x80000000


class FastNumPyCanvas(BaseCanvas):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        w = width if width is not None else shutil.get_terminal_size().columns
        h = height if height is not None else shutil.get_terminal_size().lines
        super().__init__(w, h)

        # Pre-allocate the IO bytearray for zero-allocation renders
        self._io_buf = bytearray()

        self._init_buffers(self.width, self.height)

    def _init_buffers(self, w: int, h: int) -> None:
        # Instead of 20 distinct arrays, we use 2 contiguous structured arrays
        self.back = np.zeros((h, w), dtype=CELL_DTYPE)
        self.front = np.empty((h, w), dtype=CELL_DTYPE)

        # Guarantee initial diff triggers by maximizing the front buffer
        self.front['char'] = 0xFFFFFFFF
        self.front['fg'] = 0xFFFFFFFF
        self.front['bg'] = 0xFFFFFFFF
        self.front['style'] = 0xFF

        self.clear()
        # Generous headroom for escape codes
        self._io_buf = bytearray(w * h * 32)

    # --- Cell Inspection Implementations (API Preserved) ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return chr(self.back['char'][y, x])
        return " "

    def get_fg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if 0 <= x < self.width and 0 <= y < self.height:
            val = int(self.back['fg'][y, x])
            if val & COLOR_FLAG:
                return ((val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF)
        return None

    def get_bg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if 0 <= x < self.width and 0 <= y < self.height:
            val = int(self.back['bg'][y, x])
            if val & COLOR_FLAG:
                return ((val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF)
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.back['style'][y, x])
        return 0

    # --- Vectorized Region / Zone Inspection Overrides (API Preserved) ---
    def _clamp_region(self, x: int, y: int, w: int, h: int) -> Tuple[slice, slice]:
        x_start = max(0, x)
        x_end = min(self.width, x + w)
        y_start = max(0, y)
        y_end = min(self.height, y + h)
        return slice(y_start, y_end), slice(x_start, x_end)

    def get_region_chars(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        return np.vectorize(chr, otypes=[str])(self.back['char'][s_y, s_x])

    def get_region_fg(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        sub = self.back['fg'][s_y, s_x]

        res = np.empty(sub.shape, dtype=object)
        mask = (sub & COLOR_FLAG) != 0
        res[~mask] = None

        if np.any(mask):
            masked_vals = sub[mask]
            r = (masked_vals >> 16) & 0xFF
            g = (masked_vals >> 8) & 0xFF
            b = masked_vals & 0xFF
            res[mask] = [(int(red), int(green), int(blue))
                         for red, green, blue in zip(r, g, b)]
        return res

    def get_region_bg(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        sub = self.back['bg'][s_y, s_x]

        res = np.empty(sub.shape, dtype=object)
        mask = (sub & COLOR_FLAG) != 0
        res[~mask] = None

        if np.any(mask):
            masked_vals = sub[mask]
            r = (masked_vals >> 16) & 0xFF
            g = (masked_vals >> 8) & 0xFF
            b = masked_vals & 0xFF
            res[mask] = [(int(red), int(green), int(blue))
                         for red, green, blue in zip(r, g, b)]
        return res

    def get_region_styles(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        return self.back['style'][s_y, s_x].copy()

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
        self.back['char'].fill(32)
        self.back['fg'].fill(0)
        self.back['bg'].fill(0)
        self.back['style'].fill(0)

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

        char_ords = np.fromiter(
            (ord(c) for c in target_text), dtype=np.uint32, count=len(target_text))
        self.back['char'][y, x_start:x_end] = char_ords

        if fg is not UNSET:
            if fg is None:
                self.back['fg'][y, x_start:x_end] = 0
            else:
                self.back['fg'][y, x_start:x_end] = COLOR_FLAG | (
                    fg[0] << 16) | (fg[1] << 8) | fg[2]

        if bg is not UNSET:
            if bg is None:
                self.back['bg'][y, x_start:x_end] = 0
            else:
                self.back['bg'][y, x_start:x_end] = COLOR_FLAG | (
                    bg[0] << 16) | (bg[1] << 8) | bg[2]

        if style is not UNSET:
            self.back['style'][y, x_start:x_end] = 0 if style is None else style

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
                self.back['fg'][region] = 0
            else:
                self.back['fg'][region] = COLOR_FLAG | (
                    fg[0] << 16) | (fg[1] << 8) | fg[2]

        if bg is not UNSET:
            if bg is None:
                self.back['bg'][region] = 0
            else:
                self.back['bg'][region] = COLOR_FLAG | (
                    bg[0] << 16) | (bg[1] << 8) | bg[2]

        if style is not UNSET:
            self.back['style'][region] = 0 if style is None else style

    def _style_to_sgr(self, flags: int) -> str:
        # Kept for strict backwards compatibility if called externally,
        # but the inner loop now uses a faster byte-level equivalent.
        sgr = []
        if flags & 1:
            sgr.append("1")
        if flags & 2:
            sgr.append("2")
        if flags & 4:
            sgr.append("3")
        if flags & 8:
            sgr.append("4")
        if flags & 16:
            sgr.append("5")
        if flags & 32:
            sgr.append("7")
        return ";".join(sgr)

    def render(self) -> None:
        # 1. Instant single-pass diff in C space
        diff = self.back != self.front
        if not np.any(diff):
            return

        # 2. Vectorized 1D extraction (Extract flat arrays in C before the loop)
        y_indices, x_indices = np.where(diff)
        dirty_chars = self.back['char'][diff]
        dirty_fgs = self.back['fg'][diff]
        dirty_bgs = self.back['bg'][diff]
        dirty_styles = self.back['style'][diff]

        out = self._io_buf
        del out[:]  # Clear buffer without reallocation

        last_y, last_x = -1, -1
        cur_fg, cur_bg, cur_style = -1, -1, -1

        # 3. Zip flat scalar arrays (Zero NumPy record lookup overhead)
        for y, x, char, fg, bg, st in zip(y_indices, x_indices, dirty_chars, dirty_fgs, dirty_bgs, dirty_styles):

            # Jump cursor only if non-contiguous
            if y != last_y or x != last_x + 1:
                out.extend(f"\033[{y+1};{x+1}H".encode('ascii'))

            # State machine check
            if fg != cur_fg or bg != cur_bg or st != cur_style:
                sgr = [b"0"]

                if st > 0:
                    if st & 1:
                        sgr.append(b"1")
                    if st & 2:
                        sgr.append(b"2")
                    if st & 4:
                        sgr.append(b"3")
                    if st & 8:
                        sgr.append(b"4")
                    if st & 16:
                        sgr.append(b"5")
                    if st & 32:
                        sgr.append(b"7")

                if fg & COLOR_FLAG:
                    sgr.append(
                        f"38;2;{(fg >> 16) & 0xFF};{(fg >> 8) & 0xFF};{fg & 0xFF}".encode('ascii'))

                if bg & COLOR_FLAG:
                    sgr.append(
                        f"48;2;{(bg >> 16) & 0xFF};{(bg >> 8) & 0xFF};{bg & 0xFF}".encode('ascii'))

                out.extend(b"\033[" + b";".join(sgr) + b"m")
                cur_fg, cur_bg, cur_style = fg, bg, st

            # Direct character emit
            out.extend(chr(char).encode('utf-8'))
            last_y, last_x = y, x

        # Synchronize front buffer
        self.front[diff] = self.back[diff]

        # Zero-copy binary flush
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout.buffer.write(out)
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(out.decode('utf-8'))
            sys.stdout.flush()
