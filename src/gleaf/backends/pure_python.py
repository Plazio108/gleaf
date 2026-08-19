"""Pure Python Double-Buffered Delta Renderer integrated with TerminalCaps."""

import struct
import sys

from ..caps import RGB, Modifiers, TerminalCaps
from ..textures import (
    CELL_STRUCT_FMT,
    EMPTY_CELL,
    MODE_CLEAR,
    MODE_SET,
)
from .base import UNSET, BaseCanvas, BaseTexture

try:
    import termios
    import tty
except ImportError:
    termios = None


class PurePythonCanvas(BaseCanvas):
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        caps: TerminalCaps | None = None,
    ):
        super().__init__(width, height)

        self.caps = caps if caps is not None else TerminalCaps()

        # Double Buffering
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

        # Stateful ANSI Tracking
        self._current_fg: RGB | int | None = None
        self._current_bg: RGB | int | None = None
        self._current_style: int = -1
        self._current_ul_fg: RGB | int | None = None
        self._cursor_x: int = -1
        self._cursor_y: int = -1

    def _create_grid(self, w: int, h: int):
        return [
            [
                {
                    "char": " ",
                    "fg": None,
                    "bg": None,
                    "style": Modifiers.NORMAL,
                    "ul_fg": None,
                }
                for _ in range(w)
            ]
            for _ in range(h)
        ]

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["char"]
        return " "

    def get_fg(self, x: int, y: int) -> RGB | int | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["fg"]
        return None

    def get_bg(self, x: int, y: int) -> RGB | int | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["bg"]
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["style"]
        return Modifiers.NORMAL

    def get_ul_fg(self, x: int, y: int) -> RGB | int | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x].get("ul_fg")
        return None

    def _invalidate_front_buffer(self):
        for y in range(self.height):
            for x in range(self.width):
                self.front_buffer[y][x]["char"] = None
        self._current_fg = None
        self._current_bg = None
        self._current_style = -1
        self._current_ul_fg = None
        self._cursor_x = -1
        self._cursor_y = -1

    def resize(self, width: int, height: int):
        self.width, self.height = width, height
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)
        sys.__stdout__.write("\033[2J")
        sys.__stdout__.flush()
        self._invalidate_front_buffer()

    def enter_alternate_screen(self):
        super().enter_alternate_screen()

        self._invalidate_front_buffer()

    def exit_alternate_screen(self):
        super().exit_alternate_screen()

    def clear(self):
        empty = {
            "char": " ",
            "fg": None,
            "bg": None,
            "style": Modifiers.NORMAL,
            "ul_fg": None,
        }
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(
        self,
        x: int,
        y: int,
        text: str,
        fg=UNSET,
        bg=UNSET,
        style=UNSET,
        ul_fg=UNSET,
    ):
        if y < 0 or y >= self.height:
            return

        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                cell = self.grid[y][cx]
                cell["char"] = char
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if style is not UNSET:
                    cell["style"] = style
                if ul_fg is not UNSET:
                    cell["ul_fg"] = ul_fg

    def edit_region_colors(
        self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, ul_fg=UNSET
    ):
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if ul_fg is not UNSET:
                    cell["ul_fg"] = ul_fg

    def edit_region_style(
        self, x: int, y: int, w: int, h: int, style: int, mode: str = "add"
    ):
        """Modifies style flags for a region using add, remove, or toggle modes."""
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if mode == "add":
                    cell["style"] |= style
                elif mode == "remove":
                    cell["style"] &= ~style
                elif mode == "toggle":
                    cell["style"] ^= style
                elif mode == "set":
                    cell["style"] = style

    def render(self, compute_only=False):
        out = []
        app = out.append

        grid = self.grid
        front = self.front_buffer

        cur_fg = self._current_fg
        cur_bg = self._current_bg
        cur_style = self._current_style
        cur_ul_fg = self._current_ul_fg
        cx = self._cursor_x
        cy = self._cursor_y

        format_ansi = self.caps.format_style_ansi

        for y in range(self.height):
            row_back = grid[y]
            row_front = front[y]

            for x in range(self.width):
                cell_back = row_back[x]
                cell_front = row_front[x]

                if cell_back == cell_front:
                    continue

                row_front[x] = cell_back.copy()

                # Optimize cursor movement
                if cx != x or cy != y:
                    app(f"\033[{y + 1};{x + 1}H")

                fg = cell_back["fg"]
                bg = cell_back["bg"]
                style = cell_back["style"]
                ul_fg = cell_back.get("ul_fg")

                # Check if attributes changed
                if (
                    style != cur_style
                    or fg != cur_fg
                    or bg != cur_bg
                    or ul_fg != cur_ul_fg
                ):
                    # Reset formatting when changing styles to prevent trailing attribute bugs
                    app("\033[0m")

                    # Generate optimal ANSI escape sequence via TerminalCaps
                    ansi_code = format_ansi(fg=fg, bg=bg, style=style, ul_fg=ul_fg)
                    if ansi_code:
                        app(ansi_code)

                    cur_fg = fg
                    cur_bg = bg
                    cur_style = style
                    cur_ul_fg = ul_fg

                app(cell_back["char"])
                cx = x + 1
                cy = y

        self._current_fg = cur_fg
        self._current_bg = cur_bg
        self._current_style = cur_style
        self._current_ul_fg = cur_ul_fg
        self._cursor_x = cx
        self._cursor_y = cy

        if compute_only:
            return "".join(out)

        if out:
            sys.__stdout__.write("".join(out))
            sys.__stdout__.flush()

    def apply_texture(self, texture: "PurePythonTexture", x: int, y: int):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end = min(self.width, x + texture.width)
        cy_end = min(self.height, y + texture.height)
        if cx_start >= cx_end or cy_start >= cy_end:
            return

        tx_start, ty_start = cx_start - x, cy_start - y
        c_w, t_w = self.width, texture.width
        t_cells = texture.cells

        # Hoist locals for pointer-math speed in loop
        b_ch = self.b_char
        b_h_fg, b_fr, b_fg, b_fb = self.b_has_fg, self.b_fg_r, self.b_fg_g, self.b_fg_b
        b_h_bg, b_br, b_bg, b_bb = self.b_has_bg, self.b_bg_r, self.b_bg_g, self.b_bg_b
        b_h_ul, b_ur, b_ug, b_ub = self.b_has_ul, self.b_ul_r, self.b_ul_g, self.b_ul_b
        b_st = self.b_style

        for cy in range(cy_start, cy_end):
            ty = ty_start + (cy - cy_start)
            c_base, t_base = cy * c_w, ty * t_w

            for cx in range(cx_start, cx_end):
                tx = tx_start + (cx - cx_start)
                c_idx, t_idx = c_base + cx, t_base + tx

                ch, fr, fg, fb, fm, br, bg, bb, bm, ur, ug, ub, um, st, sm = t_cells[
                    t_idx
                ]

                if ch != 0:
                    b_ch[c_idx] = chr(ch)  # Convert binary int to char string

                if fm == MODE_SET:
                    b_h_fg[c_idx] = 1
                    b_fr[c_idx] = fr
                    b_fg[c_idx] = fg
                    b_fb[c_idx] = fb
                elif fm == MODE_CLEAR:
                    b_h_fg[c_idx] = 0

                if bm == MODE_SET:
                    b_h_bg[c_idx] = 1
                    b_br[c_idx] = br
                    b_bg[c_idx] = bg
                    b_bb[c_idx] = bb
                elif bm == MODE_CLEAR:
                    b_h_bg[c_idx] = 0

                if um == MODE_SET:
                    b_h_ul[c_idx] = 1
                    b_ur[c_idx] = ur
                    b_ug[c_idx] = ug
                    b_ub[c_idx] = ub
                elif um == MODE_CLEAR:
                    b_h_ul[c_idx] = 0

                if sm == MODE_SET:
                    b_st[c_idx] = st
                elif sm == MODE_CLEAR:
                    b_st[c_idx] = 0


class PurePythonTexture(BaseTexture):
    def __init__(self, width: int, height: int, data_buffer=None):
        super().__init__(width, height)
        if data_buffer is not None:
            self.cells = list(struct.iter_unpack(CELL_STRUCT_FMT, data_buffer))
        else:
            self.cells = [EMPTY_CELL] * (width * height)

    def clear(self):
        self.cells = [EMPTY_CELL] * (self.width * self.height)

    # --- Getters ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= y < self.height and 0 <= x < self.width:
            ch = self.cells[y * self.width + x][0]
            return chr(ch) if ch != 0 else " "
        return " "

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            cell = self.cells[y * self.width + x]
            if cell[4] == MODE_SET:
                return (cell[1], cell[2], cell[3])
        return None

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= y < self.height and 0 <= x < self.width:
            cell = self.cells[y * self.width + x]
            if cell[8] == MODE_SET:
                return (cell[5], cell[6], cell[7])
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= y < self.height and 0 <= x < self.width:
            cell = self.cells[y * self.width + x]
            if cell[14] == MODE_SET:
                return cell[13]
        return 0

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

        f_mode = (
            MODE_SET
            if fg not in (UNSET, None)
            else (MODE_CLEAR if fg is None else None)
        )
        b_mode = (
            MODE_SET
            if bg not in (UNSET, None)
            else (MODE_CLEAR if bg is None else None)
        )
        u_mode = (
            MODE_SET
            if ul_fg not in (UNSET, None)
            else (MODE_CLEAR if ul_fg is None else None)
        )
        s_mode = (
            MODE_SET
            if style not in (UNSET, None, 0)
            else (MODE_CLEAR if style in (None, 0) else None)
        )

        fr, fg_c, fb = fg if f_mode == MODE_SET else (0, 0, 0)
        br, bg_c, bb = bg if b_mode == MODE_SET else (0, 0, 0)
        ur, ug_c, ub = ul_fg if u_mode == MODE_SET else (0, 0, 0)
        st = style if s_mode == MODE_SET else 0

        idx_base = y * self.width
        cells = self.cells

        for i, char in enumerate(chars_to_draw):
            idx = idx_base + cx_start + i
            (
                c_ch,
                c_fr,
                c_fg,
                c_fb,
                c_fm,
                c_br,
                c_bg,
                c_bb,
                c_bm,
                c_ur,
                c_ug,
                c_ub,
                c_um,
                c_st,
                c_sm,
            ) = cells[idx]

            n_ch = ord(char)
            n_fr, n_fg, n_fb, n_fm = (
                (fr, fg_c, fb, f_mode)
                if f_mode is not None
                else (c_fr, c_fg, c_fb, c_fm)
            )
            n_br, n_bg, n_bb, n_bm = (
                (br, bg_c, bb, b_mode)
                if b_mode is not None
                else (c_br, c_bg, c_bb, c_bm)
            )
            n_ur, n_ug, n_ub, n_um = (
                (ur, ug_c, ub, u_mode)
                if u_mode is not None
                else (c_ur, c_ug, c_ub, c_um)
            )
            n_st, n_sm = (st, s_mode) if s_mode is not None else (c_st, c_sm)

            cells[idx] = (
                n_ch,
                n_fr,
                n_fg,
                n_fb,
                n_fm,
                n_br,
                n_bg,
                n_bb,
                n_bm,
                n_ur,
                n_ug,
                n_ub,
                n_um,
                n_st,
                n_sm,
            )

    # --- Zone Editing ---
    def edit_region_colors(
        self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, ul_fg=UNSET
    ):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end, cy_end = min(self.width, x + w), min(self.height, y + h)
        if cx_start >= cx_end or cy_start >= cy_end:
            return

        f_mode = (
            MODE_SET
            if fg not in (UNSET, None)
            else (MODE_CLEAR if fg is None else None)
        )
        b_mode = (
            MODE_SET
            if bg not in (UNSET, None)
            else (MODE_CLEAR if bg is None else None)
        )
        u_mode = (
            MODE_SET
            if ul_fg not in (UNSET, None)
            else (MODE_CLEAR if ul_fg is None else None)
        )

        fr, fg_c, fb = fg if f_mode == MODE_SET else (0, 0, 0)
        br, bg_c, bb = bg if b_mode == MODE_SET else (0, 0, 0)
        ur, ug_c, ub = ul_fg if u_mode == MODE_SET else (0, 0, 0)

        cells = self.cells
        width = self.width

        for cy in range(cy_start, cy_end):
            idx_base = cy * width
            for cx in range(cx_start, cx_end):
                idx = idx_base + cx
                (
                    c_ch,
                    c_fr,
                    c_fg,
                    c_fb,
                    c_fm,
                    c_br,
                    c_bg,
                    c_bb,
                    c_bm,
                    c_ur,
                    c_ug,
                    c_ub,
                    c_um,
                    c_st,
                    c_sm,
                ) = cells[idx]

                n_fr, n_fg, n_fb, n_fm = (
                    (fr, fg_c, fb, f_mode)
                    if f_mode is not None
                    else (c_fr, c_fg, c_fb, c_fm)
                )
                n_br, n_bg, n_bb, n_bm = (
                    (br, bg_c, bb, b_mode)
                    if b_mode is not None
                    else (c_br, c_bg, c_bb, c_bm)
                )
                n_ur, n_ug, n_ub, n_um = (
                    (ur, ug_c, ub, u_mode)
                    if u_mode is not None
                    else (c_ur, c_ug, c_ub, c_um)
                )

                cells[idx] = (
                    c_ch,
                    n_fr,
                    n_fg,
                    n_fb,
                    n_fm,
                    n_br,
                    n_bg,
                    n_bb,
                    n_bm,
                    n_ur,
                    n_ug,
                    n_ub,
                    n_um,
                    c_st,
                    c_sm,
                )

    def edit_region_style(
        self, x: int, y: int, w: int, h: int, style: int, mode: str = "add"
    ):
        cx_start, cy_start = max(0, x), max(0, y)
        cx_end, cy_end = min(self.width, x + w), min(self.height, y + h)
        if cx_start >= cx_end or cy_start >= cy_end:
            return

        cells = self.cells
        width = self.width

        for cy in range(cy_start, cy_end):
            idx_base = cy * width
            for cx in range(cx_start, cx_end):
                idx = idx_base + cx
                c = cells[idx]
                curr_st = c[13]

                if mode == "set":
                    n_st, n_sm = style, MODE_SET
                elif mode == "add":
                    n_st, n_sm = curr_st | style, MODE_SET
                elif mode == "remove":
                    n_st, n_sm = curr_st & ~style, c[14]
                elif mode == "toggle":
                    n_st, n_sm = curr_st ^ style, MODE_SET
                else:
                    continue

                cells[idx] = (
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    c[4],
                    c[5],
                    c[6],
                    c[7],
                    c[8],
                    c[9],
                    c[10],
                    c[11],
                    c[12],
                    n_st,
                    n_sm,
                )
