"""Ultra High-Performance Numba JIT Backend."""

import sys
import numpy as np
from numba import jit

# Assumes NumPyCanvas is in the same package or correctly imported
try:
    from .numpy_canvas import NumPyCanvas
except ImportError:
    from gleaf import NumPyCanvas

from typing import Optional


def warmup_numba_jit(canvas: Optional['NumbaCanvas'] = None) -> None:
    """Pre-compiles Numba JIT kernels to eliminate first-frame compilation stutter."""
    if canvas is not None:
        canvas.put_str(0, 0, "🔥", fg=(255, 128, 0), bg=(
            30, 30, 30), style=63, ul_fg=(255, 0, 0))
        _numba_render_kernel(
            canvas.b_char, canvas.b_has_fg, canvas.b_fg_r, canvas.b_fg_g, canvas.b_fg_b,
            canvas.b_has_bg, canvas.b_bg_r, canvas.b_bg_g, canvas.b_bg_b,
            canvas.b_has_ul, canvas.b_ul_r, canvas.b_ul_g, canvas.b_ul_b, canvas.b_style,
            canvas.f_char, canvas.f_has_fg, canvas.f_fg_r, canvas.f_fg_g, canvas.f_fg_b,
            canvas.f_has_bg, canvas.f_bg_r, canvas.f_bg_g, canvas.f_bg_b,
            canvas.f_has_ul, canvas.f_ul_r, canvas.f_ul_g, canvas.f_ul_b, canvas.f_style,
            canvas._out_buf
        )
        canvas.clear()
        return

    # Standalone warmup using minimal dummy arrays
    h, w = 2, 2
    b_char = np.full((h, w), 0x1F525, dtype=np.uint32)

    b_has_fg = np.ones((h, w), dtype=np.uint8)
    b_fg_r, b_fg_g, b_fg_b = (np.full((h, w), v, dtype=np.uint8)
                              for v in (255, 128, 64))

    b_has_bg = np.ones((h, w), dtype=np.uint8)
    b_bg_r, b_bg_g, b_bg_b = (np.full((h, w), v, dtype=np.uint8)
                              for v in (16, 32, 48))

    b_has_ul = np.ones((h, w), dtype=np.uint8)
    b_ul_r, b_ul_g, b_ul_b = (np.full((h, w), v, dtype=np.uint8)
                              for v in (255, 0, 0))

    b_style = np.full((h, w), 63, dtype=np.uint32)

    f_char = np.zeros((h, w), dtype=np.uint32)
    f_has_fg, f_fg_r, f_fg_g, f_fg_b = (
        np.zeros((h, w), dtype=np.uint8) for _ in range(4))
    f_has_bg, f_bg_r, f_bg_g, f_bg_b = (
        np.zeros((h, w), dtype=np.uint8) for _ in range(4))
    f_has_ul, f_ul_r, f_ul_g, f_ul_b = (
        np.zeros((h, w), dtype=np.uint8) for _ in range(4))
    f_style = np.zeros((h, w), dtype=np.uint32)

    out_buf = np.zeros(1024, dtype=np.uint8)

    _numba_render_kernel(
        b_char, b_has_fg, b_fg_r, b_fg_g, b_fg_b,
        b_has_bg, b_bg_r, b_bg_g, b_bg_b,
        b_has_ul, b_ul_r, b_ul_g, b_ul_b, b_style,
        f_char, f_has_fg, f_fg_r, f_fg_g, f_fg_b,
        f_has_bg, f_bg_r, f_bg_g, f_bg_b,
        f_has_ul, f_ul_r, f_ul_g, f_ul_b, f_style,
        out_buf
    )


@jit(nopython=True, fastmath=True, cache=True)
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


@jit(nopython=True, fastmath=True, cache=True)
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


@jit(nopython=True, fastmath=True, cache=True)
def _numba_render_kernel(
    b_char, b_has_fg, b_fg_r, b_fg_g, b_fg_b,
    b_has_bg, b_bg_r, b_bg_g, b_bg_b,
    b_has_ul, b_ul_r, b_ul_g, b_ul_b, b_style,
    f_char, f_has_fg, f_fg_r, f_fg_g, f_fg_b,
    f_has_bg, f_bg_r, f_bg_g, f_bg_b,
    f_has_ul, f_ul_r, f_ul_g, f_ul_b, f_style,
    out_buf
):
    h, w = b_char.shape
    pos = 0
    last_y, last_x = -1, -1

    # Active Span State Machine
    curr_st = 0xFFFFFFFF
    curr_bhfg = 255
    curr_fg_r, curr_fg_g, curr_fg_b = 0, 0, 0
    curr_bhbg = 255
    curr_bg_r, curr_bg_g, curr_bg_b = 0, 0, 0
    curr_bhul = 255
    curr_ul_r, curr_ul_g, curr_ul_b = 0, 0, 0

    for y in range(h):
        for x in range(w):
            bc = b_char[y, x]
            bhfg = b_has_fg[y, x]
            bhbg = b_has_bg[y, x]
            bhul = b_has_ul[y, x]
            bst = b_style[y, x]

            # Delta comparison
            diff = (
                bc != f_char[y, x] or
                bst != f_style[y, x] or
                bhfg != f_has_fg[y, x] or
                bhbg != f_has_bg[y, x] or
                bhul != f_has_ul[y, x] or
                (bhfg == 1 and (b_fg_r[y, x] != f_fg_r[y, x] or b_fg_g[y, x] != f_fg_g[y, x] or b_fg_b[y, x] != f_fg_b[y, x])) or
                (bhbg == 1 and (b_bg_r[y, x] != f_bg_r[y, x] or b_bg_g[y, x] != f_bg_g[y, x] or b_bg_b[y, x] != f_bg_b[y, x])) or
                (bhul == 1 and (b_ul_r[y, x] != f_ul_r[y, x] or b_ul_g[y, x]
                 != f_ul_g[y, x] or b_ul_b[y, x] != f_ul_b[y, x]))
            )

            if not diff:
                continue

            is_contiguous = (y == last_y) and (x == last_x + 1)

            # Check if attributes perfectly match the currently active terminal span
            attrs_match = (bst == curr_st and bhfg ==
                           curr_bhfg and bhbg == curr_bhbg and bhul == curr_bhul)

            if attrs_match:
                if bhfg == 1 and (b_fg_r[y, x] != curr_fg_r or b_fg_g[y, x] != curr_fg_g or b_fg_b[y, x] != curr_fg_b):
                    attrs_match = False
                elif bhbg == 1 and (b_bg_r[y, x] != curr_bg_r or b_bg_g[y, x] != curr_bg_g or b_bg_b[y, x] != curr_bg_b):
                    attrs_match = False
                elif bhul == 1 and (b_ul_r[y, x] != curr_ul_r or b_ul_g[y, x] != curr_ul_g or b_ul_b[y, x] != curr_ul_b):
                    attrs_match = False

            if is_contiguous and attrs_match:
                # SPAN BATCHING: Same style/colors as previous char.
                # Just emit the UTF-8 bytes and skip all ANSI resets!
                pos = _append_utf8(out_buf, pos, bc)
            else:
                # NEW SPAN: Needs SGR generation
                if not is_contiguous:
                    out_buf[pos] = 27
                    out_buf[pos + 1] = 91  # "\033["
                    pos = _append_int(out_buf, pos + 2, y + 1)
                    out_buf[pos] = 59  # ';'
                    pos = _append_int(out_buf, pos + 1, x + 1)
                    out_buf[pos] = 72  # 'H'
                    pos += 1

                # SGR Reset "\033[0"
                out_buf[pos] = 27
                out_buf[pos + 1] = 91
                out_buf[pos + 2] = 48
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
                    # ";38;2;"
                    out_buf[pos] = 59
                    out_buf[pos + 1] = 51
                    out_buf[pos + 2] = 56
                    out_buf[pos + 3] = 59
                    out_buf[pos + 4] = 50
                    out_buf[pos + 5] = 59
                    pos = _append_int(out_buf, pos + 6, b_fg_r[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_fg_g[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_fg_b[y, x])

                # Background Truecolor
                if bhbg == 1:
                    # ";48;2;"
                    out_buf[pos] = 59
                    out_buf[pos + 1] = 52
                    out_buf[pos + 2] = 56
                    out_buf[pos + 3] = 59
                    out_buf[pos + 4] = 50
                    out_buf[pos + 5] = 59
                    pos = _append_int(out_buf, pos + 6, b_bg_r[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_bg_g[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_bg_b[y, x])

                # Underline Truecolor
                if bhul == 1:
                    # ";58;2;"
                    out_buf[pos] = 59
                    out_buf[pos + 1] = 53
                    out_buf[pos + 2] = 56
                    out_buf[pos + 3] = 59
                    out_buf[pos + 4] = 50
                    out_buf[pos + 5] = 59
                    pos = _append_int(out_buf, pos + 6, b_ul_r[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_ul_g[y, x])
                    out_buf[pos] = 59
                    pos += 1
                    pos = _append_int(out_buf, pos, b_ul_b[y, x])

                out_buf[pos] = 109  # 'm'
                pos += 1

                # Append character bytes
                pos = _append_utf8(out_buf, pos, bc)

                # Update the active span state tracker
                curr_st = bst
                curr_bhfg = bhfg
                if bhfg == 1:
                    curr_fg_r, curr_fg_g, curr_fg_b = b_fg_r[y,
                                                             x], b_fg_g[y, x], b_fg_b[y, x]
                curr_bhbg = bhbg
                if bhbg == 1:
                    curr_bg_r, curr_bg_g, curr_bg_b = b_bg_r[y,
                                                             x], b_bg_g[y, x], b_bg_b[y, x]
                curr_bhul = bhul
                if bhul == 1:
                    curr_ul_r, curr_ul_g, curr_ul_b = b_ul_r[y,
                                                             x], b_ul_g[y, x], b_ul_b[y, x]

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
            f_has_ul[y, x] = bhul
            f_ul_r[y, x] = b_ul_r[y, x]
            f_ul_g[y, x] = b_ul_g[y, x]
            f_ul_b[y, x] = b_ul_b[y, x]
            f_style[y, x] = bst

            last_y, last_x = y, x

    return pos


@jit(nopython=True, fastmath=True, cache=True)
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
        self._out_buf = np.zeros(self.width * self.height * 64, dtype=np.uint8)

    def _init_buffers(self, w: int, h: int) -> None:
        super()._init_buffers(w, h)
        self._out_buf = np.zeros(w * h * 64, dtype=np.uint8)

    def get_region_fg_buffers(self, x: int, y: int, w: int, h: int):
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
            self.b_has_bg, self.b_bg_r, self.b_bg_g, self.b_bg_b,
            self.b_has_ul, self.b_ul_r, self.b_ul_g, self.b_ul_b, self.b_style,
            self.f_char, self.f_has_fg, self.f_fg_r, self.f_fg_g, self.f_fg_b,
            self.f_has_bg, self.f_bg_r, self.f_bg_g, self.f_bg_b,
            self.f_has_ul, self.f_ul_r, self.f_ul_g, self.f_ul_b, self.f_style,
            self._out_buf
        )

        if bytes_written > 0:
            sys.stdout.buffer.write(self._out_buf[:bytes_written].tobytes())
            sys.stdout.buffer.flush()
