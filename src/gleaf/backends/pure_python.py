"""Pure Python Double-Buffered Delta Renderer integrated with TerminalCaps."""

import sys

from ..caps import RGB, Modifiers, TerminalCaps
from .base import BaseCanvas

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
        fg=None,
        bg=None,
        style=Modifiers.NORMAL,
        ul_fg=None,
    ):
        if y < 0 or y >= self.height:
            return

        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                cell = self.grid[y][cx]
                cell["char"] = char
                if fg is not None:
                    cell["fg"] = fg
                if bg is not None:
                    cell["bg"] = bg
                if style is not None:
                    cell["style"] = style
                if ul_fg is not None:
                    cell["ul_fg"] = ul_fg

    def edit_region_colors(
        self, x: int, y: int, w: int, h: int, fg=None, bg=None, ul_fg=None
    ):
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not None:
                    cell["fg"] = fg
                if bg is not None:
                    cell["bg"] = bg
                if ul_fg is not None:
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
