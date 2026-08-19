"""High-Performance NumPy Backend with Vectorized Frame Buffering."""

import shutil
import sys

import numpy as np

from ..textures import MODE_CLEAR, MODE_SET, TEXTURE_DTYPE
from .base import UNSET, BaseCanvas, BaseTexture

try:
    from ..caps import Modifiers, TerminalCaps
except ImportError:
    from gleaf.caps import Modifiers, TerminalCaps


try:
    import termios
    import tty
except ImportError:
    termios = None


class NumPyCanvas(BaseCanvas):
    def __init__(self, width: int | None = None, height: int | None = None):
        w = width if width is not None else shutil.get_terminal_size().columns
        h = height if height is not None else shutil.get_terminal_size().lines
        super().__init__(w, h)

        self.caps = TerminalCaps()
        self._style_cache = {}

        # --- Prebind capabilities to eliminate hasattr overhead in hot loops ---
        if hasattr(self.caps, "modifiers_to_sgr"):
            self._mod_func = self.caps.modifiers_to_sgr
        elif hasattr(self.caps, "get_modifier_sgr"):
            self._mod_func = self.caps.get_modifier_sgr
        else:
            self._mod_func = None

        self._fg_func = (
            self.caps.sgr_fg
            if hasattr(self.caps, "sgr_fg")
            else (self.caps.fg_sgr if hasattr(self.caps, "fg_sgr") else None)
        )
        self._bg_func = (
            self.caps.sgr_bg
            if hasattr(self.caps, "sgr_bg")
            else (self.caps.bg_sgr if hasattr(self.caps, "bg_sgr") else None)
        )
        self._ul_func = (
            self.caps.sgr_ul
            if hasattr(self.caps, "sgr_ul")
            else (self.caps.ul_sgr if hasattr(self.caps, "ul_sgr") else None)
        )

        self._init_buffers(self.width, self.height)

    def _init_buffers(self, w: int, h: int) -> None:
        """Allocates backend struct-of-arrays memory layout."""
        # Back buffers
        self.b_char = np.full((h, w), 32, dtype=np.uint32)
        self.b_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_has_fg = np.zeros((h, w), dtype=np.uint8)

        self.b_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.b_bg_b = np.zeros((h, w), dtype=np.uint8)
        self.b_has_bg = np.zeros((h, w), dtype=np.uint8)

        self.b_ul_r = np.zeros((h, w), dtype=np.uint8)
        self.b_ul_g = np.zeros((h, w), dtype=np.uint8)
        self.b_ul_b = np.zeros((h, w), dtype=np.uint8)
        self.b_has_ul = np.zeros((h, w), dtype=np.uint8)

        self.b_style = np.zeros((h, w), dtype=np.uint32)

        # Front buffers
        self.f_char = np.full((h, w), 0xFFFFFFFF, dtype=np.uint32)
        self.f_fg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_fg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_has_fg = np.full((h, w), 255, dtype=np.uint8)

        self.f_bg_r = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_g = np.zeros((h, w), dtype=np.uint8)
        self.f_bg_b = np.zeros((h, w), dtype=np.uint8)
        self.f_has_bg = np.full((h, w), 255, dtype=np.uint8)

        self.f_ul_r = np.zeros((h, w), dtype=np.uint8)
        self.f_ul_g = np.zeros((h, w), dtype=np.uint8)
        self.f_ul_b = np.zeros((h, w), dtype=np.uint8)
        self.f_has_ul = np.full((h, w), 255, dtype=np.uint8)

        self.f_style = np.full((h, w), 0xFFFFFFFF, dtype=np.uint32)

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return chr(self.b_char[y, x])
        return " "

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height and self.b_has_fg[y, x]:
            return (
                int(self.b_fg_r[y, x]),
                int(self.b_fg_g[y, x]),
                int(self.b_fg_b[y, x]),
            )
        return None

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height and self.b_has_bg[y, x]:
            return (
                int(self.b_bg_r[y, x]),
                int(self.b_bg_g[y, x]),
                int(self.b_bg_b[y, x]),
            )
        return None

    def get_ul_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height and self.b_has_ul[y, x]:
            return (
                int(self.b_ul_r[y, x]),
                int(self.b_ul_g[y, x]),
                int(self.b_ul_b[y, x]),
            )
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.b_style[y, x])
        return 0

    # --- Vectorized Region / Zone Inspection Overrides ---
    def _clamp_region(self, x: int, y: int, w: int, h: int) -> tuple[slice, slice]:
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
        sub_r, sub_g, sub_b = (
            self.b_fg_r[s_y, s_x],
            self.b_fg_g[s_y, s_x],
            self.b_fg_b[s_y, s_x],
        )

        res = np.empty(sub_has.shape, dtype=object)
        mask = sub_has == 1
        res[~mask] = None
        if np.any(mask):
            rgb_tuples = list(zip(sub_r[mask], sub_g[mask], sub_b[mask]))
            res[mask] = [(int(r), int(g), int(b)) for r, g, b in rgb_tuples]
        return res

    def get_region_bg(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        s_y, s_x = self._clamp_region(x, y, w, h)
        sub_has = self.b_has_bg[s_y, s_x]
        sub_r, sub_g, sub_b = (
            self.b_bg_r[s_y, s_x],
            self.b_bg_g[s_y, s_x],
            self.b_bg_b[s_y, s_x],
        )

        res = np.empty(sub_has.shape, dtype=object)
        mask = sub_has == 1
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

    def clear(self) -> None:
        self.b_char.fill(32)
        self.b_has_fg.fill(0)
        self.b_has_bg.fill(0)
        self.b_has_ul.fill(0)
        self.b_style.fill(0)

    def put_str(
        self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET
    ) -> None:
        if y < 0 or y >= self.height or not text:
            return

        # --- FAST PATH: Single-character writes (Matrix / TUI text drawing) ---
        if len(text) == 1:
            if 0 <= x < self.width:
                self.b_char[y, x] = ord(text)
                if fg is not UNSET:
                    if fg is None:
                        self.b_has_fg[y, x] = 0
                    else:
                        self.b_has_fg[y, x] = 1
                        self.b_fg_r[y, x], self.b_fg_g[y, x], self.b_fg_b[y, x] = fg
                if bg is not UNSET:
                    if bg is None:
                        self.b_has_bg[y, x] = 0
                    else:
                        self.b_has_bg[y, x] = 1
                        self.b_bg_r[y, x], self.b_bg_g[y, x], self.b_bg_b[y, x] = bg
                if ul_fg is not UNSET:
                    if ul_fg is None:
                        self.b_has_ul[y, x] = 0
                    else:
                        self.b_has_ul[y, x] = 1
                        self.b_ul_r[y, x], self.b_ul_g[y, x], self.b_ul_b[y, x] = ul_fg
                if style is not UNSET:
                    self.b_style[y, x] = 0 if style is None else style
            return

        x_start = max(0, x)
        x_end = min(self.width, x + len(text))
        if x_start >= x_end:
            return

        text_offset_start = x_start - x
        text_offset_end = text_offset_start + (x_end - x_start)
        target_text = text[text_offset_start:text_offset_end]

        char_ords = np.fromiter(
            (ord(c) for c in target_text), dtype=np.uint32, count=len(target_text)
        )
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

        if ul_fg is not UNSET:
            if ul_fg is None:
                self.b_has_ul[y, x_start:x_end] = 0
            else:
                self.b_has_ul[y, x_start:x_end] = 1
                self.b_ul_r[y, x_start:x_end] = ul_fg[0]
                self.b_ul_g[y, x_start:x_end] = ul_fg[1]
                self.b_ul_b[y, x_start:x_end] = ul_fg[2]

        if style is not UNSET:
            self.b_style[y, x_start:x_end] = 0 if style is None else style

    def edit_region_colors(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        fg=UNSET,
        bg=UNSET,
        style=UNSET,
        ul_fg=UNSET,
    ) -> None:
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

        if ul_fg is not UNSET:
            if ul_fg is None:
                self.b_has_ul[region] = 0
            else:
                self.b_has_ul[region] = 1
                self.b_ul_r[region] = ul_fg[0]
                self.b_ul_g[region] = ul_fg[1]
                self.b_ul_b[region] = ul_fg[2]

        if style is not UNSET:
            self.b_style[region] = 0 if style is None else style

    def _style_to_sgr(self, flags: int) -> str:
        """Translates style bitmask to ANSI SGR string via prebound/memoized cache."""
        if flags in self._style_cache:
            return self._style_cache[flags]

        if self._mod_func is not None:
            sgr = self._mod_func(flags)
        else:
            sgr_list = []
            if flags & Modifiers.BOLD:
                sgr_list.append("1")
            if flags & Modifiers.DIM:
                sgr_list.append("2")
            if flags & Modifiers.ITALIC:
                sgr_list.append("3")

            if flags & Modifiers.CURLY_UNDERLINE:
                sgr_list.append("4:3")
            elif flags & Modifiers.DOTTED_UNDERLINE:
                sgr_list.append("4:4")
            elif flags & Modifiers.DASHED_UNDERLINE:
                sgr_list.append("4:5")
            elif flags & Modifiers.DOUBLE_UNDERLINE:
                sgr_list.append("4:2")
            elif flags & Modifiers.UNDERLINE:
                sgr_list.append("4")

            if flags & Modifiers.BLINK:
                sgr_list.append("5")
            if flags & (Modifiers.REVERSE | Modifiers.STANDOUT):
                sgr_list.append("7")
            if flags & Modifiers.STRIKETHROUGH:
                sgr_list.append("9")
            if flags & Modifiers.OVERLINE:
                sgr_list.append("53")
            if flags & Modifiers.HIDDEN:
                sgr_list.append("8")

            sgr = ";".join(sgr_list)

        self._style_cache[flags] = sgr
        return sgr

    def _fg_sgr(self, r: int, g: int, b: int) -> str:
        if self._fg_func is not None:
            return self._fg_func(r, g, b)
        return f"38;2;{r};{g};{b}"

    def _bg_sgr(self, r: int, g: int, b: int) -> str:
        if self._bg_func is not None:
            return self._bg_func(r, g, b)
        return f"48;2;{r};{g};{b}"

    def _ul_sgr(self, r: int, g: int, b: int) -> str:
        if self._ul_func is not None:
            return self._ul_func(r, g, b)
        return f"58;2;{r};{g};{b}"

    # def render(self, compute_only=False):
    #     diff = (
    #         (self.b_char != self.f_char)
    #         | (self.b_style != self.f_style)
    #         | (self.b_has_fg != self.f_has_fg)
    #         | (self.b_has_bg != self.f_has_bg)
    #         | (self.b_has_ul != self.f_has_ul)
    #         | (
    #             (self.b_has_fg == 1)
    #             & (
    #                 (self.b_fg_r != self.f_fg_r)
    #                 | (self.b_fg_g != self.f_fg_g)
    #                 | (self.b_fg_b != self.f_fg_b)
    #             )
    #         )
    #         | (
    #             (self.b_has_bg == 1)
    #             & (
    #                 (self.b_bg_r != self.f_bg_r)
    #                 | (self.b_bg_g != self.f_bg_g)
    #                 | (self.b_bg_b != self.f_bg_b)
    #             )
    #         )
    #         | (
    #             (self.b_has_ul == 1)
    #             & (
    #                 (self.b_ul_r != self.f_ul_r)
    #                 | (self.b_ul_g != self.f_ul_g)
    #                 | (self.b_ul_b != self.f_ul_b)
    #             )
    #         )
    #     )

    #     if not np.any(diff):
    #         return

    #     y_indices, x_indices = np.where(diff)
    #     buf = []
    #     app = buf.append

    #     i = 0
    #     n = len(y_indices)

    #     while i < n:
    #         y, x = y_indices[i], x_indices[i]
    #         app(f"\033[{y + 1};{x + 1}H")

    #         st = self.b_style[y, x]
    #         has_fg, fg_r, fg_g, fg_b = (
    #             self.b_has_fg[y, x],
    #             self.b_fg_r[y, x],
    #             self.b_fg_g[y, x],
    #             self.b_fg_b[y, x],
    #         )
    #         has_bg, bg_r, bg_g, bg_b = (
    #             self.b_has_bg[y, x],
    #             self.b_bg_r[y, x],
    #             self.b_bg_g[y, x],
    #             self.b_bg_b[y, x],
    #         )
    #         has_ul, ul_r, ul_g, ul_b = (
    #             self.b_has_ul[y, x],
    #             self.b_ul_r[y, x],
    #             self.b_ul_g[y, x],
    #             self.b_ul_b[y, x],
    #         )

    #         sgr_codes = ["0"]
    #         if st > 0:
    #             sgr = self._style_to_sgr(st)
    #             if sgr:
    #                 sgr_codes.append(sgr)

    #         if has_fg:
    #             sgr_codes.append(self._fg_sgr(fg_r, fg_g, fg_b))

    #         if has_bg:
    #             sgr_codes.append(self._bg_sgr(bg_r, bg_g, bg_b))

    #         if has_ul:
    #             sgr_codes.append(self._ul_sgr(ul_r, ul_g, ul_b))

    #         app(f"\033[{';'.join(sgr_codes)}m")

    #         chars = []
    #         while i < n and y_indices[i] == y and x_indices[i] == x:
    #             chars.append(chr(self.b_char[y, x]))
    #             i += 1
    #             if i < n and y_indices[i] == y and x_indices[i] == x + 1:
    #                 nx = x_indices[i]
    #                 if (
    #                     self.b_style[y, nx] != st
    #                     or self.b_has_fg[y, nx] != has_fg
    #                     or self.b_has_bg[y, nx] != has_bg
    #                     or self.b_has_ul[y, nx] != has_ul
    #                     or (
    #                         has_fg
    #                         and (
    #                             self.b_fg_r[y, nx] != fg_r
    #                             or self.b_fg_g[y, nx] != fg_g
    #                             or self.b_fg_b[y, nx] != fg_b
    #                         )
    #                     )
    #                     or (
    #                         has_bg
    #                         and (
    #                             self.b_bg_r[y, nx] != bg_r
    #                             or self.b_bg_g[y, nx] != bg_g
    #                             or self.b_bg_b[y, nx] != bg_b
    #                         )
    #                     )
    #                     or (
    #                         has_ul
    #                         and (
    #                             self.b_ul_r[y, nx] != ul_r
    #                             or self.b_ul_g[y, nx] != ul_g
    #                             or self.b_ul_b[y, nx] != ul_b
    #                         )
    #                     )
    #                 ):
    #                     break
    #                 x = nx

    #         app("".join(chars))

    #     # Synchronize front buffer
    #     self.f_char[diff] = self.b_char[diff]
    #     self.f_has_fg[diff] = self.b_has_fg[diff]
    #     self.f_fg_r[diff] = self.b_fg_r[diff]
    #     self.f_fg_g[diff] = self.b_fg_g[diff]
    #     self.f_fg_b[diff] = self.b_fg_b[diff]
    #     self.f_has_bg[diff] = self.b_has_bg[diff]
    #     self.f_bg_r[diff] = self.b_bg_r[diff]
    #     self.f_bg_g[diff] = self.b_bg_g[diff]
    #     self.f_bg_b[diff] = self.b_bg_b[diff]
    #     self.f_has_ul[diff] = self.b_has_ul[diff]
    #     self.f_ul_r[diff] = self.b_ul_r[diff]
    #     self.f_ul_g[diff] = self.b_ul_g[diff]
    #     self.f_ul_b[diff] = self.b_ul_b[diff]
    #     self.f_style[diff] = self.b_style[diff]

    #     if compute_only:
    #         return "".join(buf)

    #     if buf:
    #         sys.stdout.write("".join(buf))
    #         sys.stdout.flush()

    def render(self, compute_only=False):
        # 1. Vectorized Diff (Your existing fast logic)
        diff = (
            (self.b_char != self.f_char)
            | (self.b_style != self.f_style)
            | (self.b_has_fg != self.f_has_fg)
            | (self.b_has_bg != self.f_has_bg)
            | (self.b_has_ul != self.f_has_ul)
            | (
                (self.b_has_fg == 1)
                & (
                    (self.b_fg_r != self.f_fg_r)
                    | (self.b_fg_g != self.f_fg_g)
                    | (self.b_fg_b != self.f_fg_b)
                )
            )
            | (
                (self.b_has_bg == 1)
                & (
                    (self.b_bg_r != self.f_bg_r)
                    | (self.b_bg_g != self.f_bg_g)
                    | (self.b_bg_b != self.f_bg_b)
                )
            )
            | (
                (self.b_has_ul == 1)
                & (
                    (self.b_ul_r != self.f_ul_r)
                    | (self.b_ul_g != self.f_ul_g)
                    | (self.b_ul_b != self.f_ul_b)
                )
            )
        )

        if not np.any(diff):
            return

        # 2. Vectorized Extraction
        # Pull ALL changed data into 1D arrays instantly, then convert to fast native Python lists
        y_indices, x_indices = np.where(diff)

        c_vals = self.b_char[diff].tolist()
        s_vals = self.b_style[diff].tolist()

        hf_vals = self.b_has_fg[diff].tolist()
        fr_vals, fg_vals, fb_vals = (
            self.b_fg_r[diff].tolist(),
            self.b_fg_g[diff].tolist(),
            self.b_fg_b[diff].tolist(),
        )

        hb_vals = self.b_has_bg[diff].tolist()
        br_vals, bg_vals, bb_vals = (
            self.b_bg_r[diff].tolist(),
            self.b_bg_g[diff].tolist(),
            self.b_bg_b[diff].tolist(),
        )

        hu_vals = self.b_has_ul[diff].tolist()
        ur_vals, ug_vals, ub_vals = (
            self.b_ul_r[diff].tolist(),
            self.b_ul_g[diff].tolist(),
            self.b_ul_b[diff].tolist(),
        )

        buf = []
        app = buf.append

        cur_y, cur_x = -1, -1
        cur_state = None

        # 3. The Fast Loop
        # zip() runs at C-speed. We are now dealing exclusively with native Python integers.
        iterator = zip(
            y_indices,
            x_indices,
            c_vals,
            s_vals,
            hf_vals,
            fr_vals,
            fg_vals,
            fb_vals,
            hb_vals,
            br_vals,
            bg_vals,
            bb_vals,
            hu_vals,
            ur_vals,
            ug_vals,
            ub_vals,
        )

        for y, x, c, s, hf, fr, fg, fb, hb, br, bg, bb, hu, ur, ug, ub in iterator:
            # We bundle all style data into a tuple.
            # Python compares tuples instantly, replacing your massive 14-check if statement.
            state = (s, hf, fr, fg, fb, hb, br, bg, bb, hu, ur, ug, ub)

            # Move cursor only if we aren't continuing from the exact previous cell
            if y != cur_y or x != cur_x + 1:
                app(f"\033[{y + 1};{x + 1}H")

            # Apply ANSI sequences only if the styling actually changed
            if state != cur_state:
                sgr_codes = ["0"]
                if s > 0:
                    sgr = self._style_to_sgr(s)
                    if sgr:
                        sgr_codes.append(sgr)
                if hf:
                    sgr_codes.append(self._fg_sgr(fr, fg, fb))
                if hb:
                    sgr_codes.append(self._bg_sgr(br, bg, bb))
                if hu:
                    sgr_codes.append(self._ul_sgr(ur, ug, ub))

                app(f"\033[{';'.join(sgr_codes)}m")
                cur_state = state

            app(chr(c))
            cur_y, cur_x = y, x

        # 4. Synchronize front buffer (Your existing fast logic)
        self.f_char[diff] = self.b_char[diff]
        self.f_has_fg[diff] = self.b_has_fg[diff]
        self.f_fg_r[diff] = self.b_fg_r[diff]
        self.f_fg_g[diff] = self.b_fg_g[diff]
        self.f_fg_b[diff] = self.b_fg_b[diff]
        self.f_has_bg[diff] = self.b_has_bg[diff]
        self.f_bg_r[diff] = self.b_bg_r[diff]
        self.f_bg_g[diff] = self.b_bg_g[diff]
        self.f_bg_b[diff] = self.b_bg_b[diff]
        self.f_has_ul[diff] = self.b_has_ul[diff]
        self.f_ul_r[diff] = self.b_ul_r[diff]
        self.f_ul_g[diff] = self.b_ul_g[diff]
        self.f_ul_b[diff] = self.b_ul_b[diff]
        self.f_style[diff] = self.b_style[diff]

        if compute_only:
            return "".join(buf)

        if buf:
            sys.stdout.write("".join(buf))
            sys.stdout.flush()

    def apply_texture(self, texture, x: int, y: int):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end = min(self.width, x + texture.width)
        cy_end = min(self.height, y + texture.height)

        if cx_start >= cx_end or cy_start >= cy_end:
            return

        tx_start, ty_start = cx_start - x, cy_start - y
        c_w, t_w = self.width, texture.width
        t_cells = texture.cells

        # Hoist array references explicitly to avoid tuple unpacking shadow bugs
        b_char = self.b_char
        b_h_fg = self.b_has_fg
        b_fg_r = self.b_fg_r
        b_fg_g = self.b_fg_g
        b_fg_b = self.b_fg_b

        b_h_bg = self.b_has_bg
        b_bg_r = self.b_bg_r
        b_bg_g = self.b_bg_g
        b_bg_b = self.b_bg_b

        b_h_ul = self.b_has_ul
        b_ul_r = self.b_ul_r
        b_ul_g = self.b_ul_g
        b_ul_b = self.b_ul_b

        b_style = self.b_style

        # Use hardcoded integers to bypass any `MODE_SET` import evaluation bugs
        MODE_SET = 1
        MODE_CLEAR = 2

        for cy in range(cy_start, cy_end):
            ty = ty_start + (cy - cy_start)
            c_base, t_base = cy * c_w, ty * t_w

            for cx in range(cx_start, cx_end):
                tx = tx_start + (cx - cx_start)
                c_idx = c_base + cx
                t_idx = t_base + tx

                # Direct index access prevents the 15-variable unpack from
                # shifting values or shadowing local aliases.
                cell = t_cells[t_idx]

                # 0: Character
                ch = cell[0]
                if ch != 0:
                    b_char[c_idx] = chr(ch)

                # Foreground Mode (index 4)
                fm = cell[4]
                if fm == MODE_SET:
                    b_h_fg[c_idx] = 1
                    b_fg_r[c_idx] = cell[1]
                    b_fg_g[c_idx] = cell[2]
                    b_fg_b[c_idx] = cell[3]
                elif fm == MODE_CLEAR:
                    b_h_fg[c_idx] = 0

                # Background Mode (index 8)
                bm = cell[8]
                if bm == MODE_SET:
                    b_h_bg[c_idx] = 1
                    b_bg_r[c_idx] = cell[5]
                    b_bg_g[c_idx] = cell[6]
                    b_bg_b[c_idx] = cell[7]
                elif bm == MODE_CLEAR:
                    b_h_bg[c_idx] = 0

                # Underline Mode (index 12)
                um = cell[12]
                if um == MODE_SET:
                    b_h_ul[c_idx] = 1
                    b_ul_r[c_idx] = cell[9]
                    b_ul_g[c_idx] = cell[10]
                    b_ul_b[c_idx] = cell[11]
                elif um == MODE_CLEAR:
                    b_h_ul[c_idx] = 0

                # Style Mode (index 14)
                sm = cell[14]
                if sm == MODE_SET:
                    b_style[c_idx] = cell[13]
                elif sm == MODE_CLEAR:
                    b_style[c_idx] = 0


class NumpyTexture(BaseTexture):
    def __init__(self, width: int, height: int, data_buffer=None):
        super().__init__(width, height)
        if data_buffer is not None:
            self.data = np.frombuffer(data_buffer, dtype=TEXTURE_DTYPE).reshape(
                (height, width)
            )
            if not self.data.flags.writeable:
                self.data = self.data.copy()
        else:
            self.data = np.zeros((height, width), dtype=TEXTURE_DTYPE)

    def clear(self):
        self.data[...] = 0

    # --- Getters ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= y < self.height and 0 <= x < self.width:
            code = self.data["char"][y, x]
            return chr(code) if code != 0 else " "
        return " "

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            if self.data["fg_mode"][y, x] == MODE_SET:
                return (
                    int(self.data["fg_r"][y, x]),
                    int(self.data["fg_g"][y, x]),
                    int(self.data["fg_b"][y, x]),
                )
        return None

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            if self.data["bg_mode"][y, x] == MODE_SET:
                return (
                    int(self.data["bg_r"][y, x]),
                    int(self.data["bg_g"][y, x]),
                    int(self.data["bg_b"][y, x]),
                )
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= y < self.height and 0 <= x < self.width:
            if self.data["style_mode"][y, x] == MODE_SET:
                return int(self.data["style"][y, x])
        return 0

    # --- Vectorized Region Getters ---
    def get_region_chars(self, x: int, y: int, w: int, h: int) -> list[list[str]]:
        sub = self.data["char"][
            max(0, y) : min(self.height, y + h), max(0, x) : min(self.width, x + w)
        ]
        return [[chr(c) if c != 0 else " " for c in row] for row in sub]

    def get_region_styles(self, x: int, y: int, w: int, h: int) -> list[list[int]]:
        sub = self.data[
            max(0, y) : min(self.height, y + h), max(0, x) : min(self.width, x + w)
        ]
        return [
            [
                int(cell["style"]) if cell["style_mode"] == MODE_SET else 0
                for cell in row
            ]
            for row in sub
        ]

    # --- Drawing API ---
    def put_str(
        self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET
    ):
        if y < 0 or y >= self.height or x >= self.width:
            return

        cx_start = max(0, x)
        text_start = cx_start - x
        avail = self.width - cx_start
        chars_to_draw = text[text_start : text_start + avail]
        if not chars_to_draw:
            return
        cx_end = cx_start + len(chars_to_draw)

        self.data["char"][y, cx_start:cx_end] = [ord(c) for c in chars_to_draw]

        if fg is not UNSET:
            if fg is None:
                self.data["fg_mode"][y, cx_start:cx_end] = MODE_CLEAR
            else:
                self.data["fg_r"][y, cx_start:cx_end] = fg[0]
                self.data["fg_g"][y, cx_start:cx_end] = fg[1]
                self.data["fg_b"][y, cx_start:cx_end] = fg[2]
                self.data["fg_mode"][y, cx_start:cx_end] = MODE_SET

        if bg is not UNSET:
            if bg is None:
                self.data["bg_mode"][y, cx_start:cx_end] = MODE_CLEAR
            else:
                self.data["bg_r"][y, cx_start:cx_end] = bg[0]
                self.data["bg_g"][y, cx_start:cx_end] = bg[1]
                self.data["bg_b"][y, cx_start:cx_end] = bg[2]
                self.data["bg_mode"][y, cx_start:cx_end] = MODE_SET

        if ul_fg is not UNSET:
            if ul_fg is None:
                self.data["ul_mode"][y, cx_start:cx_end] = MODE_CLEAR
            else:
                self.data["ul_r"][y, cx_start:cx_end] = ul_fg[0]
                self.data["ul_g"][y, cx_start:cx_end] = ul_fg[1]
                self.data["ul_b"][y, cx_start:cx_end] = ul_fg[2]
                self.data["ul_mode"][y, cx_start:cx_end] = MODE_SET

        if style is not UNSET:
            if style in (None, 0):
                self.data["style_mode"][y, cx_start:cx_end] = MODE_CLEAR
            else:
                self.data["style"][y, cx_start:cx_end] = style
                self.data["style_mode"][y, cx_start:cx_end] = MODE_SET

    # --- Vectorized Zone Editing ---
    def edit_region_colors(
        self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, ul_fg=UNSET
    ):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end, cy_end = min(self.width, x + w), min(self.height, y + h)
        if cx_start >= cx_end or cy_start >= cy_end:
            return

        sub = self.data[cy_start:cy_end, cx_start:cx_end]

        if fg is not UNSET:
            if fg is None:
                sub["fg_mode"] = MODE_CLEAR
            else:
                sub["fg_r"], sub["fg_g"], sub["fg_b"] = fg
                sub["fg_mode"] = MODE_SET

        if bg is not UNSET:
            if bg is None:
                sub["bg_mode"] = MODE_CLEAR
            else:
                sub["bg_r"], sub["bg_g"], sub["bg_b"] = bg
                sub["bg_mode"] = MODE_SET

        if ul_fg is not UNSET:
            if ul_fg is None:
                sub["ul_mode"] = MODE_CLEAR
            else:
                sub["ul_r"], sub["ul_g"], sub["ul_b"] = ul_fg
                sub["ul_mode"] = MODE_SET

    def edit_region_style(
        self, x: int, y: int, w: int, h: int, style: int, mode: str = "add"
    ):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end, cy_end = min(self.width, x + w), min(self.height, y + h)
        if cx_start >= cx_end or cy_start >= cy_end:
            return

        sub = self.data[cy_start:cy_end, cx_start:cx_end]

        if mode == "set":
            sub["style"] = style
            sub["style_mode"] = MODE_SET
        elif mode == "add":
            sub["style"] |= np.uint32(style)
            sub["style_mode"] = MODE_SET
        elif mode == "remove":
            sub["style"] &= ~np.uint32(style)
        elif mode == "toggle":
            sub["style"] ^= np.uint32(style)
            sub["style_mode"] = MODE_SET
