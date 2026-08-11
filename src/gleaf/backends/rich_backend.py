"""High-Performance Rich Backend with Single-Pass Frame Buffering."""

import sys
from typing import Optional, Tuple
from rich.console import Console
from rich.style import Style
from .base import BaseCanvas, UNSET


class RichCanvas(BaseCanvas):
    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        console: Optional[Console] = None
    ):
        self.console = console or Console()

        # Default to Console dimensions if not explicitly passed
        w = width if width is not None else self.console.width
        h = height if height is not None else self.console.height

        super().__init__(w, h)
        self._style_cache = {}

        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w: int, h: int):
        return [
            [{"char": " ", "fg": None, "bg": None, "style": 0}
                for _ in range(w)]
            for _ in range(h)
        ]

    def _get_style(self, fg: Optional[Tuple[int, int, int]], bg: Optional[Tuple[int, int, int]], style_flags: int) -> Optional[Style]:
        key = (fg, bg, style_flags)
        if key in self._style_cache:
            return self._style_cache[key]

        color = f"rgb({fg[0]},{fg[1]},{fg[2]})" if fg else None
        bgcolor = f"rgb({bg[0]},{bg[1]},{bg[2]})" if bg else None

        bold = bool(style_flags & 1)
        dim = bool(style_flags & 2)
        italic = bool(style_flags & 4)
        underline = bool(style_flags & 8)
        blink = bool(style_flags & 16)
        reverse = bool(style_flags & 32)

        style = Style(
            color=color,
            bgcolor=bgcolor,
            bold=bold if bold else None,
            dim=dim if dim else None,
            italic=italic if italic else None,
            underline=underline if underline else None,
            blink=blink if blink else None,
            reverse=reverse if reverse else None,
        ) if (color or bgcolor or style_flags) else None

        self._style_cache[key] = style
        return style

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["char"]
        return " "

    def get_fg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["fg"]
        return None

    def get_bg(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
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

        # Invalidate front buffer to force a total redraw on next frame
        self.front_buffer = [
            [{"char": None, "fg": None, "bg": None, "style": None}
                for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def auto_resize(self) -> bool:
        cw, ch = self.console.width, self.console.height
        if cw != self.width or ch != self.height:
            self.resize(cw, ch)
            return True
        return False

    def enter_alternate_screen(self) -> None:
        self.console.show_cursor(False)
        self.console.file.write("\033[?1049h\033[H\033[2J")
        self.console.file.flush()

    def exit_alternate_screen(self) -> None:
        self.console.file.write("\033[0m\033[?1049l")
        self.console.show_cursor(True)
        self.console.file.flush()

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
                if char is not UNSET:
                    cell["char"] = char
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if style is not UNSET:
                    cell["style"] = style

    def edit_region_colors(self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, style=UNSET) -> None:
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if style is not UNSET:
                    cell["style"] = style

    def render(self) -> None:
        buf = []
        app = buf.append

        for y in range(self.height):
            row = self.grid[y]
            front_row = self.front_buffer[y]

            x = 0
            while x < self.width:
                if row[x] == front_row[x]:
                    x += 1
                    continue

                app(f"\033[{y+1};{x+1}H")

                curr_style_data = (row[x]["fg"], row[x]["bg"], row[x]["style"])
                char_buffer = []

                while (
                    x < self.width
                    and (row[x]["fg"], row[x]["bg"], row[x]["style"]) == curr_style_data
                    and row[x] != front_row[x]
                ):
                    char_buffer.append(row[x]["char"])
                    front_row[x] = row[x].copy()
                    x += 1

                text = "".join(char_buffer)
                style_obj = self._get_style(*curr_style_data)

                if style_obj:
                    app(style_obj.render(text))
                else:
                    app(text)

        if buf:
            self.console.file.write("".join(buf))
            self.console.file.flush()
