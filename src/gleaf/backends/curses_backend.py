"""Curses Backend with Native Bitmask Attribute Mapping & Terminal Management."""

import curses

try:
    from ..caps import Modifiers, TerminalCaps
except ImportError:
    from gleaf.caps import Modifiers, TerminalCaps

from .base import UNSET, BaseCanvas


class CursesCanvas(BaseCanvas):
    def __init__(
        self, stdscr=None, width: int | None = None, height: int | None = None
    ):
        self.stdscr = stdscr

        # Infer dimensions from stdscr if available
        if self.stdscr is not None and (width is None or height is None):
            max_y, max_x = self.stdscr.getmaxyx()
            width = width if width is not None else max_x
            height = height if height is not None else max_y

        super().__init__(width or 80, height or 24)

        self.caps = TerminalCaps(
            has_truecolor=False, has_256color=False, has_extended_underline=False
        )

        self._color_pair_cache = {}
        self._next_pair_id = 1

        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w: int, h: int):
        return [
            [
                {"char": " ", "fg": None, "bg": None, "style": 0, "ul_fg": None}
                for _ in range(w)
            ]
            for _ in range(h)
        ]

    def _snap_to_6x6x6(self, rgb: tuple[int, int, int]) -> int:
        r, g, b = rgb
        r_i = round(r / 255 * 5)
        g_i = round(g / 255 * 5)
        b_i = round(b / 255 * 5)
        return 16 + (36 * r_i) + (6 * g_i) + b_i

    def _get_color_pair(
        self, fg: tuple[int, int, int] | None, bg: tuple[int, int, int] | None
    ) -> int:
        if not curses.has_colors():
            return 0

        fg_idx = self._snap_to_6x6x6(fg) if fg else -1
        bg_idx = self._snap_to_6x6x6(bg) if bg else -1
        key = (fg_idx, bg_idx)

        if key in self._color_pair_cache:
            return self._color_pair_cache[key]

        if self._next_pair_id < getattr(curses, "COLOR_PAIRS", 256):
            pair_id = self._next_pair_id
            curses.init_pair(pair_id, fg_idx, bg_idx)
            self._color_pair_cache[key] = pair_id
            self._next_pair_id += 1
            return pair_id

        return 0

    def _map_curses_attrs(self, style_flags: int) -> int:
        """Map Modifiers bitmask flags to curses attributes with graceful fallbacks."""
        if not style_flags:
            return 0

        attrs = 0

        # Standard Curses attributes
        if style_flags & Modifiers.BOLD:
            attrs |= curses.A_BOLD
        if style_flags & Modifiers.DIM:
            attrs |= getattr(curses, "A_DIM", 0)
        if style_flags & Modifiers.ITALIC:
            attrs |= getattr(curses, "A_ITALIC", 0)
        if style_flags & Modifiers.UNDERLINE:
            attrs |= curses.A_UNDERLINE
        if style_flags & Modifiers.REVERSE:
            attrs |= curses.A_REVERSE
        if style_flags & Modifiers.STANDOUT:
            attrs |= getattr(curses, "A_STANDOUT", curses.A_REVERSE)
        if style_flags & Modifiers.BLINK:
            attrs |= curses.A_BLINK
        if style_flags & Modifiers.HIDDEN:
            attrs |= getattr(curses, "A_INVIS", 0)

        # Extended Underlines -> Fallback to standard Curses Underline
        ext_underline_mask = (
            Modifiers.DOUBLE_UNDERLINE
            | Modifiers.CURLY_UNDERLINE
            | Modifiers.DOTTED_UNDERLINE
            | Modifiers.DASHED_UNDERLINE
        )
        if style_flags & ext_underline_mask:
            attrs |= curses.A_UNDERLINE

        # Strikethrough fallback (available in modern ncurses builds)
        if style_flags & Modifiers.STRIKETHROUGH:
            attrs |= getattr(curses, "A_STRIKETHROUGH", 0)

        return attrs

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["char"]
        return " "

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["fg"]
        return None

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["bg"]
        return None

    def get_style(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["style"]
        return 0

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width == self.width and new_height == self.height:
            return

        self.width = new_width
        self.height = new_height
        self.grid = self._create_grid(self.width, self.height)

        self.front_buffer = [
            [
                {"char": None, "fg": None, "bg": None, "style": None, "ul_fg": None}
                for _ in range(self.width)
            ]
            for _ in range(self.height)
        ]

    def auto_resize(self) -> bool:
        if self.stdscr is None:
            return False
        max_y, max_x = self.stdscr.getmaxyx()
        if max_x != self.width or max_y != self.height:
            self.resize(max_x, max_y)
            return True
        return False

    def enter_alternate_screen(self) -> None:
        if self.stdscr is None:
            self.stdscr = curses.initscr()

        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.stdscr.keypad(True)

        has_colors = curses.has_colors()
        if has_colors:
            curses.start_color()
            curses.use_default_colors()

        self.caps = TerminalCaps(
            has_truecolor=False, has_256color=has_colors, has_extended_underline=False
        )

        max_y, max_x = self.stdscr.getmaxyx()
        self.resize(max_x, max_y)

    def exit_alternate_screen(self) -> None:
        if self.stdscr is not None:
            self.stdscr.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.curs_set(1)
            curses.endwin()

    def clear(self) -> None:
        empty = {"char": " ", "fg": None, "bg": None, "style": 0, "ul_fg": None}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(
        self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET
    ) -> None:
        if y < 0 or y >= self.height:
            return
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                cell = self.grid[y][cx]
                if char is not UNSET:
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
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if style is not UNSET:
                    cell["style"] = style
                if ul_fg is not UNSET:
                    cell["ul_fg"] = ul_fg

    def render(self, compute_only=False):
        if compute_only:
            return self._render_sequence()

        if self.stdscr is None:
            return

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell == self.front_buffer[y][x]:
                    continue

                pair_idx = self._get_color_pair(cell["fg"], cell["bg"])
                attr = curses.color_pair(pair_idx) | self._map_curses_attrs(
                    cell["style"]
                )

                try:
                    self.stdscr.addstr(y, x, cell["char"], attr)
                except curses.error:
                    # Prevents crashes when writing to the bottom-right corner cell
                    pass

                self.front_buffer[y][x] = cell.copy()

        self.stdscr.refresh()

    def _render_sequence(self):
        out = []
        app = out.append

        grid = self.grid
        front = self.front_buffer

        cur_fg = None
        cur_bg = None
        cur_style = -1
        cur_ul_fg = None
        cx = -1
        cy = -1

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

        return "".join(out)
