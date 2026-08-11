"""Curses Backend with Xterm-256 Color Quantization and Delta Rendering."""

import curses
from .base import BaseCanvas


def _snap_to_6x6x6(val):
    """Maps a 0-255 RGB value to the standard 0-5 Xterm cube index."""
    if val < 48:
        return 0
    if val < 115:
        return 1
    if val < 155:
        return 2
    if val < 195:
        return 3
    if val < 235:
        return 4
    return 5


def rgb_to_xterm256(r, g, b):
    """Calculates the closest Xterm-256 color index without heavy math."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round(((r - 8) / 247) * 24) + 232
    return 16 + (36 * _snap_to_6x6x6(r)) + (6 * _snap_to_6x6x6(g)) + _snap_to_6x6x6(b)


class CursesCanvas(BaseCanvas):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self.stdscr = None

        # Double Buffering
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

        # Curses Palette Manager
        self._color_pairs = {}
        self._next_pair_id = 1
        self._max_pairs = 256

    def _create_grid(self, w, h):
        return [
            [{'char': ' ', 'fg': None, 'bg': None} for _ in range(w)]
            for _ in range(h)
        ]

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        curses.resizeterm(height, width)
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)
        self.stdscr.clear()

    def _get_color_pair(self, fg, bg):
        if fg is None and bg is None:
            return 0

        fg_idx = rgb_to_xterm256(*fg) if fg else -1
        bg_idx = rgb_to_xterm256(*bg) if bg else -1

        key = (fg_idx, bg_idx)
        if key in self._color_pairs:
            return self._color_pairs[key]

        if self._next_pair_id < self._max_pairs:
            pair_id = self._next_pair_id
            curses.init_pair(pair_id, fg_idx, bg_idx)
            self._color_pairs[key] = pair_id
            self._next_pair_id += 1
            return pair_id

        # Fallback if we somehow exhaust terminal color pairs
        return 0

    def enter_alternate_screen(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        self._max_pairs = min(curses.COLOR_PAIRS - 1, 32767)
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.stdscr.clear()

    def exit_alternate_screen(self):
        if self.stdscr:
            curses.curs_set(1)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.stdscr = None

    def clear(self):
        empty = {'char': ' ', 'fg': None, 'bg': None}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(self, x, y, text, fg=None, bg=None, style=0):
        if y < 0 or y >= self.height:
            return
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                self.grid[y][cx].update({'char': char, 'fg': fg, 'bg': bg})

    def edit_region_colors(self, x, y, w, h, fg=None, bg=None):
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                if fg:
                    self.grid[cy][cx]['fg'] = fg
                if bg:
                    self.grid[cy][cx]['bg'] = bg

    def render(self):
        if not self.stdscr:
            return

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                front = self.front_buffer[y][x]

                if cell == front:
                    continue

                # Commit to front buffer
                front.update(cell)

                pair_id = self._get_color_pair(cell['fg'], cell['bg'])
                try:
                    self.stdscr.addstr(
                        y, x, cell['char'], curses.color_pair(pair_id))
                except curses.error:
                    pass  # Writing to bottom-right corner throws in curses

        self.stdscr.refresh()
