"""Curses Backend with Native Bitmask Attribute Mapping & Terminal Management."""

import curses
from typing import Optional, Tuple
from .base import BaseCanvas, UNSET


class CursesCanvas(BaseCanvas):
    def __init__(
        self, 
        stdscr=None, 
        width: Optional[int] = None, 
        height: Optional[int] = None
    ):
        self.stdscr = stdscr
        
        # Infer dimensions from stdscr if available
        if self.stdscr is not None and (width is None or height is None):
            max_y, max_x = self.stdscr.getmaxyx()
            width = width if width is not None else max_x
            height = height if height is not None else max_y

        super().__init__(width or 80, height or 24)

        self._color_pair_cache = {}
        self._next_pair_id = 1

        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w: int, h: int):
        return [
            [{"char": " ", "fg": None, "bg": None, "style": 0} for _ in range(w)] 
            for _ in range(h)
        ]

    def _snap_to_6x6x6(self, rgb: Tuple[int, int, int]) -> int:
        r, g, b = rgb
        r_i = round(r / 255 * 5)
        g_i = round(g / 255 * 5)
        b_i = round(b / 255 * 5)
        return 16 + (36 * r_i) + (6 * g_i) + b_i

    def _get_color_pair(self, fg: Optional[Tuple[int, int, int]], bg: Optional[Tuple[int, int, int]]) -> int:
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
        attrs = 0
        if style_flags & 1:  attrs |= curses.A_BOLD
        if style_flags & 2:  attrs |= getattr(curses, "A_DIM", 0)
        if style_flags & 4:  attrs |= getattr(curses, "A_ITALIC", 0)
        if style_flags & 8:  attrs |= curses.A_UNDERLINE
        if style_flags & 16: attrs |= curses.A_BLINK
        if style_flags & 32: attrs |= curses.A_REVERSE
        return attrs

    def resize(self, new_width: int, new_height: int) -> None:
        if new_width == self.width and new_height == self.height:
            return

        self.width = new_width
        self.height = new_height
        self.grid = self._create_grid(self.width, self.height)

        self.front_buffer = [
            [{"char": None, "fg": None, "bg": None, "style": None} for _ in range(self.width)]
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

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()

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
        empty = {"char": " ", "fg": None, "bg": None, "style": 0}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET) -> None:
        if y < 0 or y >= self.height:
            return
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                cell = self.grid[y][cx]
                if char is not UNSET: cell["char"] = char
                if fg is not UNSET: cell["fg"] = fg
                if bg is not UNSET: cell["bg"] = bg
                if style is not UNSET: cell["style"] = style

    def edit_region_colors(self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, style=UNSET) -> None:
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not UNSET: cell["fg"] = fg
                if bg is not UNSET: cell["bg"] = bg
                if style is not UNSET: cell["style"] = style

    def render(self) -> None:
        if self.stdscr is None:
            return

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell == self.front_buffer[y][x]:
                    continue

                pair_idx = self._get_color_pair(cell["fg"], cell["bg"])
                attr = curses.color_pair(pair_idx) | self._map_curses_attrs(cell["style"])

                try:
                    self.stdscr.addstr(y, x, cell["char"], attr)
                except curses.error:
                    # Prevents crashes when writing to the bottom-right corner cell
                    pass

                self.front_buffer[y][x] = cell.copy()

        self.stdscr.refresh()
