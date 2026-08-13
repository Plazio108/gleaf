"""High-Performance Row-Streaming Canvas Backend with Numba Buffer Synchronization."""

import sys
import shutil
import numpy as np
from numba import njit, prange
from typing import Optional

try:
    from ..caps import Modifiers, TerminalCaps
except ImportError:
    from gleaf.caps import Modifiers, TerminalCaps

from .numpy_backend import NumPyCanvas
from .base import UNSET

def warmup_numba_jit(*a,**k):
    pass


@njit(parallel=True, fastmath=True)
def _numba_sync_and_detect_dirty(
    b_char, b_has_fg, b_fg_r, b_fg_g, b_fg_b,
    b_has_bg, b_bg_r, b_bg_g, b_bg_b,
    b_has_ul, b_ul_r, b_ul_g, b_ul_b, b_style,
    f_char, f_has_fg, f_fg_r, f_fg_g, f_fg_b,
    f_has_bg, f_bg_r, f_bg_g, f_bg_b,
    f_has_ul, f_ul_r, f_ul_g, f_ul_b, f_style,
    explicit_row_dirty, row_is_changed
):
    h, w = b_char.shape
    for y in prange(h):
        changed = explicit_row_dirty[y]
        for x in range(w):
            bc = b_char[y, x]
            bst = b_style[y, x]
            bhfg = b_has_fg[y, x]
            bhbg = b_has_bg[y, x]
            bhul = b_has_ul[y, x]

            fc = f_char[y, x]
            fst = f_style[y, x]
            fhfg = f_has_fg[y, x]
            fhbg = f_has_bg[y, x]
            fhul = f_has_ul[y, x]

            cell_changed = False
            if bc != fc or bst != fst or bhfg != fhfg or bhbg != fhbg or bhul != fhul:
                cell_changed = True
            elif bhfg == 1 and (b_fg_r[y, x] != f_fg_r[y, x] or b_fg_g[y, x] != f_fg_g[y, x] or b_fg_b[y, x] != f_fg_b[y, x]):
                cell_changed = True
            elif bhbg == 1 and (b_bg_r[y, x] != f_bg_r[y, x] or b_bg_g[y, x] != f_bg_g[y, x] or b_bg_b[y, x] != f_bg_b[y, x]):
                cell_changed = True
            elif bhul == 1 and (b_ul_r[y, x] != f_ul_r[y, x] or b_ul_g[y, x] != f_ul_g[y, x] or b_ul_b[y, x] != f_ul_b[y, x]):
                cell_changed = True

            if cell_changed:
                changed = True
                f_char[y, x] = bc
                f_style[y, x] = bst
                f_has_fg[y, x] = bhfg
                if bhfg == 1:
                    f_fg_r[y, x] = b_fg_r[y, x]
                    f_fg_g[y, x] = b_fg_g[y, x]
                    f_fg_b[y, x] = b_fg_b[y, x]
                f_has_bg[y, x] = bhbg
                if bhbg == 1:
                    f_bg_r[y, x] = b_bg_r[y, x]
                    f_bg_g[y, x] = b_bg_g[y, x]
                    f_bg_b[y, x] = b_bg_b[y, x]
                f_has_ul[y, x] = bhul
                if bhul == 1:
                    f_ul_r[y, x] = b_ul_r[y, x]
                    f_ul_g[y, x] = b_ul_g[y, x]
                    f_ul_b[y, x] = b_ul_b[y, x]

        row_is_changed[y] = changed
        explicit_row_dirty[y] = False


class NumbaCanvas(NumPyCanvas):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        super().__init__(width, height)
        self._warmup_numba()

    def _init_buffers(self, w: int, h: int) -> None:
        super()._init_buffers(w, h)
        self._row_dirty = np.ones(h, dtype=np.bool_)
        self._row_is_changed = np.zeros(h, dtype=np.bool_)

    def _warmup_numba(self) -> None:
        h, w = 2, 2
        bc = np.full((h, w), 32, dtype=np.uint32)
        fc = np.full((h, w), 0xFFFFFFFF, dtype=np.uint32)
        zero_8 = np.zeros((h, w), dtype=np.uint8)
        zero_32 = np.zeros((h, w), dtype=np.uint32)
        dirty = np.ones(h, dtype=np.bool_)
        changed = np.zeros(h, dtype=np.bool_)
        
        _numba_sync_and_detect_dirty(
            bc, zero_8, zero_8, zero_8, zero_8,
            zero_8, zero_8, zero_8, zero_8,
            zero_8, zero_8, zero_8, zero_8, zero_32,
            fc, zero_8, zero_8, zero_8, zero_8,
            zero_8, zero_8, zero_8, zero_8,
            zero_8, zero_8, zero_8, zero_8, zero_32,
            dirty, changed
        )

    def clear(self) -> None:
        super().clear()
        self._row_dirty[:] = True

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width == self.width and new_height == self.height:
            return
        super().resize(new_width, new_height)
        self._row_dirty[:] = True

    def put_str(self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET) -> None:
        super().put_str(x, y, text, fg=fg, bg=bg, style=style, ul_fg=ul_fg)
        if 0 <= y < self.height:
            self._row_dirty[y] = True

    def edit_region_colors(self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET) -> None:
        super().edit_region_colors(x, y, w, h, fg=fg, bg=bg, style=style, ul_fg=ul_fg)
        y_start = max(0, min(self.height, y))
        y_end = max(0, min(self.height, y + h))
        if y_start < y_end:
            self._row_dirty[y_start:y_end] = True

    def render(self) -> None:
        _numba_sync_and_detect_dirty(
            self.b_char, self.b_has_fg, self.b_fg_r, self.b_fg_g, self.b_fg_b,
            self.b_has_bg, self.b_bg_r, self.b_bg_g, self.b_bg_b,
            self.b_has_ul, self.b_ul_r, self.b_ul_g, self.b_ul_b, self.b_style,
            self.f_char, self.f_has_fg, self.f_fg_r, self.f_fg_g, self.f_fg_b,
            self.f_has_bg, self.f_bg_r, self.f_bg_g, self.f_bg_b,
            self.f_has_ul, self.f_ul_r, self.f_ul_g, self.f_ul_b, self.f_style,
            self._row_dirty, self._row_is_changed
        )

        if not np.any(self._row_is_changed):
            return

        io_buf = bytearray()
        
        # Cache SGR combinations per frame to avoid redundant string joins
        sgr_cache = {}

        for y in range(self.height):
            if not self._row_is_changed[y]:
                continue

            # Position cursor at the beginning of the dirty row once
            io_buf.extend(f"\033[{y+1};1H".encode('ascii'))
            
            last_sgr_str = None
            
            for x in range(self.width):
                st = self.f_style[y, x]
                has_fg = self.f_has_fg[y, x]
                fg_r = self.f_fg_r[y, x]
                fg_g = self.f_fg_g[y, x]
                fg_b = self.f_fg_b[y, x]

                has_bg = self.f_has_bg[y, x]
                bg_r = self.f_bg_r[y, x]
                bg_g = self.f_bg_g[y, x]
                bg_b = self.f_bg_b[y, x]

                has_ul = self.f_has_ul[y, x]
                ul_r = self.f_ul_r[y, x]
                ul_g = self.f_ul_g[y, x]
                ul_b = self.f_ul_b[y, x]

                # Style lookup key for cache
                style_key = (st, has_fg, fg_r, fg_g, fg_b, has_bg, bg_r, bg_g, bg_b, has_ul, ul_r, ul_g, ul_b)
                
                sgr_str = sgr_cache.get(style_key)
                if sgr_str is None:
                    sgr_codes = ["0"]
                    if st > 0:
                        sgr = self._style_to_sgr(st)
                        if sgr:
                            sgr_codes.append(sgr)
                    if has_fg:
                        sgr_codes.append(self._fg_sgr(fg_r, fg_g, fg_b))
                    if has_bg:
                        sgr_codes.append(self._bg_sgr(bg_r, bg_g, bg_b))
                    if has_ul:
                        sgr_codes.append(self._ul_sgr(ul_r, ul_g, ul_b))
                    
                    sgr_str = ";".join(sgr_codes)
                    sgr_cache[style_key] = sgr_str

                # Only emit SGR sequence when style/color changes mid-row
                if sgr_str != last_sgr_str:
                    io_buf.extend(f"\033[{sgr_str}m".encode('ascii'))
                    last_sgr_str = sgr_str

                io_buf.append(self.f_char[y, x])

        if len(io_buf) > 0:
            sys.stdout.buffer.write(io_buf)
            sys.stdout.flush()
