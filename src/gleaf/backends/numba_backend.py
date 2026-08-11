"""Ultra High-Performance Numba JIT Backend."""

import sys
import numpy as np
from numba import jit
from gleaf import NumPyCanvas

from typing import Optional


def warmup_numba_jit(canvas: Optional[NumbaCanvas] = None) -> None:
    """Pre-compiles Numba JIT kernels to eliminate first-frame compilation stutter.

    Can be called either standalone before entering your event loop,
    or directly on an existing `NumbaCanvas` instance.
    """
    if canvas is not None:
        # Populate canvas with data that triggers all features (colors, styles, unicode)
        canvas.put_str(0, 0, "🔥", fg=(255, 128, 0), bg=(30, 30, 30), style=63)

        # Execute JIT kernel directly into buffer without flushing to terminal/stdout
        _numba_render_kernel(
            canvas.b_char, canvas.b_has_fg, canvas.b_fg_r, canvas.b_fg_g, canvas.b_fg_b,
            canvas.b_has_bg, canvas.b_bg_r, canvas.b_bg_g, canvas.b_bg_b, canvas.b_style,
            canvas.f_char, canvas.f_has_fg, canvas.f_fg_r, canvas.f_fg_g, canvas.f_fg_b,
            canvas.f_has_bg, canvas.f_bg_r, canvas.f_bg_g, canvas.f_bg_b, canvas.f_style,
            canvas._out_buf
        )
        canvas.clear()
        return

    # Standalone warmup using minimal dummy arrays
    h, w = 2, 2

    # Multi-byte UTF-8 character (e.g. '🔥' = 0x1F525) to trigger 4-byte UTF-8 encoding
    b_char = np.full((h, w), 0x1F525, dtype=np.uint32)

    # Enable foreground truecolor
    b_has_fg = np.ones((h, w), dtype=np.uint8)
    b_fg_r = np.full((h, w), 255, dtype=np.uint8)
    b_fg_g = np.full((h, w), 128, dtype=np.uint8)
    b_fg_b = np.full((h, w), 64, dtype=np.uint8)

    # Enable background truecolor
    b_has_bg = np.ones((h, w), dtype=np.uint8)
    b_bg_r = np.full((h, w), 16, dtype=np.uint8)
    b_bg_g = np.full((h, w), 32, dtype=np.uint8)
    b_bg_b = np.full((h, w), 48, dtype=np.uint8)

    # Bitmask 63 = 1|2|4|8|16|32 (Bold, Dim, Italic, Underline, Blink, Reverse)
    b_style = np.full((h, w), 63, dtype=np.uint8)

    # Diff-triggering front buffer (sentinel uninitialized values)
    f_char = np.zeros((h, w), dtype=np.uint32)
    f_has_fg = np.zeros((h, w), dtype=np.uint8)
    f_fg_r = np.zeros((h, w), dtype=np.uint8)
    f_fg_g = np.zeros((h, w), dtype=np.uint8)
    f_fg_b = np.zeros((h, w), dtype=np.uint8)

    f_has_bg = np.zeros((h, w), dtype=np.uint8)
    f_bg_r = np.zeros((h, w), dtype=np.uint8)
    f_bg_g = np.zeros((h, w), dtype=np.uint8)
    f_bg_b = np.zeros((h, w), dtype=np.uint8)

    f_style = np.zeros((h, w), dtype=np.uint8)

    out_buf = np.zeros(1024, dtype=np.uint8)

    # Trigger compilation pass
    _numba_render_kernel(
        b_char, b_has_fg, b_fg_r, b_fg_g, b_fg_b,
        b_has_bg, b_bg_r, b_bg_g, b_bg_b, b_style,
        f_char, f_has_fg, f_fg_r, f_fg_g, f_fg_b,
        f_has_bg, f_bg_r, f_bg_g, f_bg_b, f_style,
        out_buf
    )


@jit(nopython=True, fastmath=True)
def _append_int(buf, pos, val):
    if val == 0:
        buf[pos] = 48  # '0'
        return pos + 1
    start = pos
    while val > 0:
        buf[pos] = 48 + (val % 10)
        val //= 10
        pos += 1
    end = pos - 1
    while start < end:
        tmp = buf[start]
        buf[start] = buf[end]
        buf[end] = tmp
        start += 1
        end -= 1
    return pos


@jit(nopython=True, fastmath=True)
def _append_utf8(buf, pos, code):
    if code <= 0x7F:
        buf[pos] = code
        return pos + 1
    elif code <= 0x7FF:
        buf[pos] = 0xC0 | (code >> 6)
        buf[pos + 1] = 0x80 | (code & 0x3F)
        return pos + 2
    elif code <= 0xFFFF:
        buf[pos] = 0xE0 | (code >> 12)
        buf[pos + 1] = 0x80 | ((code >> 6) & 0x3F)
        buf[pos + 2] = 0x80 | (code & 0x3F)
        return pos + 3
    else:
        buf[pos] = 0xF0 | (code >> 18)
        buf[pos + 1] = 0x80 | ((code >> 12) & 0x3F)
        buf[pos + 2] = 0x80 | ((code >> 6) & 0x3F)
        buf[pos + 3] = 0x80 | (code & 0x3F)
        return pos + 4


@jit(nopython=True, fastmath=True)
def _numba_render_kernel(
    b_char, b_has_fg, b_fg_r, b_fg_g, b_fg_b, b_has_bg, b_bg_r, b_bg_g, b_bg_b, b_style,
    f_char, f_has_fg, f_fg_r, f_fg_g, f_fg_b, f_has_bg, f_bg_r, f_bg_g, f_bg_b, f_style,
    out_buf
):
    h, w = b_char.shape
    pos = 0

    last_y, last_x = -1, -1

    for y in range(h):
        for x in range(w):
            bc = b_char[y, x]
            bhfg = b_has_fg[y, x]
            bhbg = b_has_bg[y, x]
            bst = b_style[y, x]

            # Delta comparison
            diff = (
                bc != f_char[y, x] or
                bhfg != f_has_fg[y, x] or
                bhbg != f_has_bg[y, x] or
                bst != f_style[y, x] or
                (bhfg == 1 and (b_fg_r[y, x] != f_fg_r[y, x] or b_fg_g[y, x] != f_fg_g[y, x] or b_fg_b[y, x] != f_fg_b[y, x])) or
                (bhbg == 1 and (b_bg_r[y, x] != f_bg_r[y, x] or b_bg_g[y, x]
                 != f_bg_g[y, x] or b_bg_b[y, x] != f_bg_b[y, x]))
            )

            if not diff:
                continue

            # Move cursor if not adjacent cell in same row
            if y != last_y or x != last_x + 1:
                out_buf[pos] = 27
                out_buf[pos + 1] = 91  # "\033["
                pos = _append_int(out_buf, pos + 2, y + 1)
                out_buf[pos] = 59  # ';'
                pos = _append_int(out_buf, pos + 1, x + 1)
                out_buf[pos] = 72  # 'H'
                pos += 1

            # SGR Reset
            out_buf[pos] = 27
            out_buf[pos + 1] = 91
            out_buf[pos + 2] = 48  # "\033[0"
            pos += 3

            # Bitmask Text Style attributes
            if bst & 1:   # Bold
                out_buf[pos] = 59
                out_buf[pos + 1] = 49
                pos += 2
            if bst & 2:   # Dim
                out_buf[pos] = 59
                out_buf[pos + 1] = 50
                pos += 2
            if bst & 4:   # Italic
                out_buf[pos] = 59
                out_buf[pos + 1] = 51
                pos += 2
            if bst & 8:   # Underline
                out_buf[pos] = 59
                out_buf[pos + 1] = 52
                pos += 2
            if bst & 16:  # Blink
                out_buf[pos] = 59
                out_buf[pos + 1] = 53
                pos += 2
            if bst & 32:  # Reverse
                out_buf[pos] = 59
                out_buf[pos + 1] = 55
                pos += 2

            # Foreground Truecolor
            if bhfg == 1:
                out_buf[pos] = 59
                out_buf[pos + 1] = 51
                out_buf[pos + 2] = 56
                out_buf[pos + 3] = 59
                out_buf[pos + 4] = 50
                out_buf[pos + 5] = 59  # ";38;2;"
                pos = _append_int(out_buf, pos + 6, b_fg_r[y, x])
                out_buf[pos] = 59
                pos += 1
                pos = _append_int(out_buf, pos, b_fg_g[y, x])
                out_buf[pos] = 59
                pos += 1
                pos = _append_int(out_buf, pos, b_fg_b[y, x])

            # Background Truecolor
            if bhbg == 1:
                out_buf[pos] = 59
                out_buf[pos + 1] = 52
                out_buf[pos + 2] = 56
                out_buf[pos + 3] = 59
                out_buf[pos + 4] = 50
                out_buf[pos + 5] = 59  # ";48;2;"
                pos = _append_int(out_buf, pos + 6, b_bg_r[y, x])
                out_buf[pos] = 59
                pos += 1
                pos = _append_int(out_buf, pos, b_bg_g[y, x])
                out_buf[pos] = 59
                pos += 1
                pos = _append_int(out_buf, pos, b_bg_b[y, x])

            out_buf[pos] = 109  # 'm'
            pos += 1

            # Append character bytes
            pos = _append_utf8(out_buf, pos, bc)

            # Update front buffer state
            f_char[y, x] = bc
            f_has_fg[y, x] = bhfg
            f_fg_r[y, x] = b_fg_r[y, x]
            f_fg_g[y, x] = b_fg_g[y, x]
            f_fg_b[y, x] = b_fg_b[y, x]
            f_has_bg[y, x] = bhbg
            f_bg_r[y, x] = b_bg_r[y, x]
            f_bg_g[y, x] = b_bg_g[y, x]
            f_bg_b[y, x] = b_bg_b[y, x]
            f_style[y, x] = bst

            last_y, last_x = y, x

    return pos


# --- JIT Region Extraction Kernels ---
@jit(nopython=True, fastmath=True)
def _numba_extract_colors(b_has, b_r, b_g, b_b, y_start, y_end, x_start, x_end, out_has, out_rgb):
    h = y_end - y_start
    w = x_end - x_start
    for y in range(h):
        for x in range(w):
            sy, sx = y_start + y, x_start + x
            has_col = b_has[sy, sx]
            out_has[y, x] = has_col
            if has_col:
                out_rgb[y, x, 0] = b_r[sy, sx]
                out_rgb[y, x, 1] = b_g[sy, sx]
                out_rgb[y, x, 2] = b_b[sy, sx]


class NumbaCanvas(NumPyCanvas):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        super().__init__(width, height)
        # Pre-allocate byte output buffer (up to ~64 bytes per cell maximum)
        self._out_buf = np.zeros(self.width * self.height * 64, dtype=np.uint8)

    def _init_buffers(self, w: int, h: int) -> None:
        super()._init_buffers(w, h)
        self._out_buf = np.zeros(w * h * 64, dtype=np.uint8)

    # --- Numba JIT Region Getter Overrides ---
    def get_region_fg_buffers(self, x: int, y: int, w: int, h: int):
        """Ultra-fast JIT extraction returning primitive NumPy buffers (has_fg, rgb_array)."""
        s_y, s_x = self._clamp_region(x, y, w, h)
        y_start, y_end = s_y.start, s_y.stop
        x_start, x_end = s_x.start, s_x.stop

        rh, rw = y_end - y_start, x_end - x_start
        out_has = np.empty((rh, rw), dtype=np.uint8)
        out_rgb = np.empty((rh, rw, 3), dtype=np.uint8)

        _numba_extract_colors(
            self.b_has_fg, self.b_fg_r, self.b_fg_g, self.b_fg_b,
            y_start, y_end, x_start, x_end, out_has, out_rgb
        )
        return out_has, out_rgb

    def get_region_bg_buffers(self, x: int, y: int, w: int, h: int):
        """Ultra-fast JIT extraction returning primitive NumPy buffers (has_bg, rgb_array)."""
        s_y, s_x = self._clamp_region(x, y, w, h)
        y_start, y_end = s_y.start, s_y.stop
        x_start, x_end = s_x.start, s_x.stop

        rh, rw = y_end - y_start, x_end - x_start
        out_has = np.empty((rh, rw), dtype=np.uint8)
        out_rgb = np.empty((rh, rw, 3), dtype=np.uint8)

        _numba_extract_colors(
            self.b_has_bg, self.b_bg_r, self.b_bg_g, self.b_bg_b,
            y_start, y_end, x_start, x_end, out_has, out_rgb
        )
        return out_has, out_rgb

    def render(self) -> None:
        bytes_written = _numba_render_kernel(
            self.b_char, self.b_has_fg, self.b_fg_r, self.b_fg_g, self.b_fg_b,
            self.b_has_bg, self.b_bg_r, self.b_bg_g, self.b_bg_b, self.b_style,
            self.f_char, self.f_has_fg, self.f_fg_r, self.f_fg_g, self.f_fg_b,
            self.f_has_bg, self.f_bg_r, self.f_bg_g, self.f_bg_b, self.f_style,
            self._out_buf
        )

        if bytes_written > 0:
            sys.stdout.buffer.write(self._out_buf[:bytes_written].tobytes())
            sys.stdout.buffer.flush()
