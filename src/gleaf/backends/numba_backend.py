"""Numba Struct-of-Arrays (SoA) Multi-Core Parallel Canvas."""

import sys
import numpy as np
from numba import jit, prange
from .base import BaseCanvas

try:
    import termios
    import tty
except ImportError:
    termios = None


@jit(nopython=True)
def _append_int(buffer, offset, val):
    if val == 0:
        buffer[offset] = 48
        return offset + 1
    temp = np.zeros(10, dtype=np.uint8)
    count = 0
    while val > 0:
        temp[count] = 48 + (val % 10)
        val //= 10
        count += 1
    for i in range(count - 1, -1, -1):
        buffer[offset] = temp[i]
        offset += 1
    return offset


@jit(nopython=True, parallel=True, cache=True)
def _render_deltas_parallel(
    height, width, 
    char_b, fg_r_b, fg_g_b, fg_b_b, bg_r_b, bg_g_b, bg_b_b, has_fg_b, has_bg_b,
    char_f, fg_r_f, fg_g_f, fg_b_f, bg_r_f, bg_g_f, bg_b_f, has_fg_f, has_bg_f,
    line_buffers, line_lengths
):
    for y in prange(height):
        buf = line_buffers[y]
        pos = 0
        
        cur_has_fg = False
        cur_has_bg = False
        cur_fg_r, cur_fg_g, cur_fg_b = 0, 0, 0
        cur_bg_r, cur_bg_g, cur_bg_b = 0, 0, 0
        cx = -1

        for x in range(width):
            if (char_b[y, x] == char_f[y, x] and
                has_fg_b[y, x] == has_fg_f[y, x] and
                has_bg_b[y, x] == has_bg_f[y, x] and
                (not has_fg_b[y, x] or (fg_r_b[y, x] == fg_r_f[y, x] and fg_g_b[y, x] == fg_g_f[y, x] and fg_b_b[y, x] == fg_b_f[y, x])) and
                (not has_bg_b[y, x] or (bg_r_b[y, x] == bg_r_f[y, x] and bg_g_b[y, x] == bg_g_f[y, x] and bg_b_b[y, x] == bg_b_f[y, x]))
            ):
                continue

            char_f[y, x] = char_b[y, x]
            has_fg_f[y, x] = has_fg_b[y, x]
            has_bg_f[y, x] = has_bg_b[y, x]
            fg_r_f[y, x] = fg_r_b[y, x]; fg_g_f[y, x] = fg_g_b[y, x]; fg_b_f[y, x] = fg_b_b[y, x]
            bg_r_f[y, x] = bg_r_b[y, x]; bg_g_f[y, x] = bg_g_b[y, x]; bg_b_f[y, x] = bg_b_b[y, x]

            if cx != x:
                buf[pos] = 27; buf[pos+1] = 91; pos += 2
                pos = _append_int(buf, pos, y + 1); buf[pos] = 59; pos += 1
                pos = _append_int(buf, pos, x + 1); buf[pos] = 72; pos += 1

            if has_fg_b[y, x] and (not cur_has_fg or cur_fg_r != fg_r_b[y, x] or cur_fg_g != fg_g_b[y, x] or cur_fg_b != fg_b_b[y, x]):
                buf[pos]=27; buf[pos+1]=91; buf[pos+2]=51; buf[pos+3]=56; buf[pos+4]=59; buf[pos+5]=50; buf[pos+6]=59; pos+=7
                pos = _append_int(buf, pos, fg_r_b[y, x]); buf[pos] = 59; pos += 1
                pos = _append_int(buf, pos, fg_g_b[y, x]); buf[pos] = 59; pos += 1
                pos = _append_int(buf, pos, fg_b_b[y, x]); buf[pos] = 109; pos += 1
                cur_has_fg = True
                cur_fg_r, cur_fg_g, cur_fg_b = fg_r_b[y, x], fg_g_b[y, x], fg_b_b[y, x]
            elif not has_fg_b[y, x] and cur_has_fg:
                buf[pos]=27; buf[pos+1]=91; buf[pos+2]=51; buf[pos+3]=57; buf[pos+4]=109; pos+=5
                cur_has_fg = False

            if has_bg_b[y, x] and (not cur_has_bg or cur_bg_r != bg_r_b[y, x] or cur_bg_g != bg_g_b[y, x] or cur_bg_b != bg_b_b[y, x]):
                buf[pos]=27; buf[pos+1]=91; buf[pos+2]=52; buf[pos+3]=56; buf[pos+4]=59; buf[pos+5]=50; buf[pos+6]=59; pos+=7
                pos = _append_int(buf, pos, bg_r_b[y, x]); buf[pos] = 59; pos += 1
                pos = _append_int(buf, pos, bg_g_b[y, x]); buf[pos] = 59; pos += 1
                pos = _append_int(buf, pos, bg_b_b[y, x]); buf[pos] = 109; pos += 1
                cur_has_bg = True
                cur_bg_r, cur_bg_g, cur_bg_b = bg_r_b[y, x], bg_g_b[y, x], bg_b_b[y, x]
            elif not has_bg_b[y, x] and cur_has_bg:
                buf[pos]=27; buf[pos+1]=91; buf[pos+2]=52; buf[pos+3]=57; buf[pos+4]=109; pos+=5
                cur_has_bg = False

            ch = char_b[y, x]
            if ch <= 0x7F:
                buf[pos] = ch
                pos += 1
            elif ch <= 0x7FF:
                buf[pos] = 192 | (ch >> 6); buf[pos+1] = 128 | (ch & 63); pos += 2
            elif ch <= 0xFFFF:
                buf[pos] = 224 | (ch >> 12); buf[pos+1] = 128 | ((ch >> 6) & 63); buf[pos+2] = 128 | (ch & 63); pos += 3
            else:
                buf[pos] = 240 | (ch >> 18); buf[pos+1] = 128 | ((ch >> 12) & 63); buf[pos+2] = 128 | ((ch >> 6) & 63); buf[pos+3] = 128 | (ch & 63); pos += 4

            cx = x + 1

        if cur_has_fg or cur_has_bg:
            buf[pos] = 27; buf[pos+1] = 91; buf[pos+2] = 48; buf[pos+3] = 109; pos += 4

        line_lengths[y] = pos


def warmup_numba_jit():
    """Silently triggers JIT compilation without writing to stdout."""
    u8 = np.zeros((1, 1), dtype=np.uint8)
    u32 = np.zeros((1, 1), dtype=np.uint32)
    b = np.zeros((1, 1), dtype=np.bool_)
    i32 = np.zeros(1, dtype=np.int32)
    _render_deltas_parallel(
        1, 1,
        u32, u8, u8, u8, u8, u8, u8, b, b,
        u32, u8, u8, u8, u8, u8, u8, b, b,
        u8, i32
    )


class NumbaCanvas(BaseCanvas):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self._allocate_soa_buffers()

    def _allocate_soa_buffers(self):
        w, h = self.width, self.height
        # Back Buffers
        self.b_char = np.full((h, w), ord(' '), dtype=np.uint32)
        self.b_has_fg = np.zeros((h, w), dtype=np.bool_)
        self.b_has_bg = np.zeros((h, w), dtype=np.bool_)
        self.b_fg_r = np.zeros((h, w), dtype=np.uint8); self.b_fg_g = np.zeros((h, w), dtype=np.uint8); self.b_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_r = np.zeros((h, w), dtype=np.uint8); self.b_bg_g = np.zeros((h, w), dtype=np.uint8); self.b_bg_b = np.zeros((h, w), dtype=np.uint8)

        # Front Buffers
        self.f_char = np.zeros((h, w), dtype=np.uint32)
        self.f_has_fg = np.zeros((h, w), dtype=np.bool_)
        self.f_has_bg = np.zeros((h, w), dtype=np.bool_)
        self.f_fg_r = np.zeros((h, w), dtype=np.uint8); self.f_fg_g = np.zeros((h, w), dtype=np.uint8); self.f_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_r = np.zeros((h, w), dtype=np.uint8); self.f_bg_g = np.zeros((h, w), dtype=np.uint8); self.f_bg_b = np.zeros((h, w), dtype=np.uint8)

        # Output Buffers
        self._line_buffers = np.zeros((h, w * 32 + 64), dtype=np.uint8)
        self._line_lengths = np.zeros(h, dtype=np.int32)

    def resize(self, width: int, height: int):
        self.width, self.height = width, height
        self._allocate_soa_buffers()
        self.invalidate_front_buffer()

    def enter_alternate_screen(self):
        sys.__stdout__.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.__stdout__.flush()
        
        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
        self.invalidate_front_buffer()

    def exit_alternate_screen(self):
        sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
        sys.__stdout__.flush()
        
        if termios is not None and hasattr(self, '_old_term') and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def invalidate_front_buffer(self):
        if hasattr(self, 'f_char'):
            self.f_char.fill(0)

    def clear(self):
        if hasattr(self, 'b_char'):
            self.b_char.fill(ord(' '))
            self.b_has_fg.fill(False)
            self.b_has_bg.fill(False)

    def put_str(self, x, y, text, fg=None, bg=None, style=0):
        if y < 0 or y >= self.height:
            return
        length = min(len(text), self.width - x)
        if length <= 0:
            return

        self.b_char[y, x:x+length] = [ord(c) for c in text[:length]]

        if fg:
            self.b_has_fg[y, x:x+length] = True
            self.b_fg_r[y, x:x+length] = fg[0]
            self.b_fg_g[y, x:x+length] = fg[1]
            self.b_fg_b[y, x:x+length] = fg[2]

        if bg:
            self.b_has_bg[y, x:x+length] = True
            self.b_bg_r[y, x:x+length] = bg[0]
            self.b_bg_g[y, x:x+length] = bg[1]
            self.b_bg_b[y, x:x+length] = bg[2]

    def edit_region_colors(self, x, y, w, h, fg=None, bg=None):
        sy = max(0, y); ey = min(self.height, y + h)
        sx = max(0, x); ex = min(self.width, x + w)
        if fg:
            self.b_has_fg[sy:ey, sx:ex] = True
            self.b_fg_r[sy:ey, sx:ex] = fg[0]
            self.b_fg_g[sy:ey, sx:ex] = fg[1]
            self.b_fg_b[sy:ey, sx:ex] = fg[2]
        if bg:
            self.b_has_bg[sy:ey, sx:ex] = True
            self.b_bg_r[sy:ey, sx:ex] = bg[0]
            self.b_bg_g[sy:ey, sx:ex] = bg[1]
            self.b_bg_b[sy:ey, sx:ex] = bg[2]

    def render(self):
        _render_deltas_parallel(
            self.height, self.width,
            self.b_char, self.b_fg_r, self.b_fg_g, self.b_fg_b, self.b_bg_r, self.b_bg_g, self.b_bg_b, self.b_has_fg, self.b_has_bg,
            self.f_char, self.f_fg_r, self.f_fg_g, self.f_fg_b, self.f_bg_r, self.f_bg_g, self.f_bg_b, self.f_has_fg, self.f_has_bg,
            self._line_buffers, self._line_lengths
        )
        
        out_chunks = [
            self._line_buffers[y, :self._line_lengths[y]].tobytes() 
            for y in range(self.height) if self._line_lengths[y] > 0
        ]
        if out_chunks:
            sys.__stdout__.buffer.write(b"".join(out_chunks))
            sys.__stdout__.buffer.flush()
