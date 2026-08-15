"""High-Performance NumPy Backend with Vectorized Frame Buffering."""

import shutil
import sys

import numpy as np

try:
    from ..caps import Modifiers, TerminalCaps
except ImportError:
    from gleaf.caps import Modifiers, TerminalCaps

from .base import UNSET, BaseCanvas

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

    def render(self, compute_only=False):
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

        y_indices, x_indices = np.where(diff)
        buf = []
        app = buf.append

        i = 0
        n = len(y_indices)

        while i < n:
            y, x = y_indices[i], x_indices[i]
            app(f"\033[{y + 1};{x + 1}H")

            st = self.b_style[y, x]
            has_fg, fg_r, fg_g, fg_b = (
                self.b_has_fg[y, x],
                self.b_fg_r[y, x],
                self.b_fg_g[y, x],
                self.b_fg_b[y, x],
            )
            has_bg, bg_r, bg_g, bg_b = (
                self.b_has_bg[y, x],
                self.b_bg_r[y, x],
                self.b_bg_g[y, x],
                self.b_bg_b[y, x],
            )
            has_ul, ul_r, ul_g, ul_b = (
                self.b_has_ul[y, x],
                self.b_ul_r[y, x],
                self.b_ul_g[y, x],
                self.b_ul_b[y, x],
            )

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

            app(f"\033[{';'.join(sgr_codes)}m")

            chars = []
            while i < n and y_indices[i] == y and x_indices[i] == x:
                chars.append(chr(self.b_char[y, x]))
                i += 1
                if i < n and y_indices[i] == y and x_indices[i] == x + 1:
                    nx = x_indices[i]
                    if (
                        self.b_style[y, nx] != st
                        or self.b_has_fg[y, nx] != has_fg
                        or self.b_has_bg[y, nx] != has_bg
                        or self.b_has_ul[y, nx] != has_ul
                        or (
                            has_fg
                            and (
                                self.b_fg_r[y, nx] != fg_r
                                or self.b_fg_g[y, nx] != fg_g
                                or self.b_fg_b[y, nx] != fg_b
                            )
                        )
                        or (
                            has_bg
                            and (
                                self.b_bg_r[y, nx] != bg_r
                                or self.b_bg_g[y, nx] != bg_g
                                or self.b_bg_b[y, nx] != bg_b
                            )
                        )
                        or (
                            has_ul
                            and (
                                self.b_ul_r[y, nx] != ul_r
                                or self.b_ul_g[y, nx] != ul_g
                                or self.b_ul_b[y, nx] != ul_b
                            )
                        )
                    ):
                        break
                    x = nx

            app("".join(chars))

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
