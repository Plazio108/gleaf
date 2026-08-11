"""High-Performance Rich Backend with Console Integration and Resizing."""

import sys
from typing import Optional
from rich.console import Console
from rich.style import Style
from .base import BaseCanvas

try:
    import termios
    import tty
except ImportError:
    termios = None


class RichCanvas(BaseCanvas):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None, console: Optional[Console] = None):
        # 1. Integrate Rich Console instance (or create default)
        self.console = console or Console()

        # 2. Default to Console size if dimensions are omitted
        w = width if width is not None else self.console.width
        h = height if height is not None else self.console.height

        super().__init__(w, h)
        self._style_cache = {}

        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w: int, h: int):
        return [
            [{"char": " ", "fg": None, "bg": None} for _ in range(w)] 
            for _ in range(h)
        ]

    def _get_style(self, fg, bg) -> Optional[Style]:
        key = (fg, bg)
        if key in self._style_cache:
            return self._style_cache[key]

        color = f"rgb({fg[0]},{fg[1]},{fg[2]})" if fg else None
        bgcolor = f"rgb({bg[0]},{bg[1]},{bg[2]})" if bg else None

        style = Style(color=color, bgcolor=bgcolor) if (color or bgcolor) else None
        self._style_cache[key] = style
        return style

    def resize(self, new_width: int, new_height: int) -> None:
        """Resizes grid buffers and invalidates front_buffer to force a clean redraw."""
        if new_width == self.width and new_height == self.height:
            return

        self.width = new_width
        self.height = new_height
        self.grid = self._create_grid(self.width, self.height)

        # Invalidate front_buffer using sentinel values so every cell updates on next render
        self.front_buffer = [
            [{"char": None, "fg": None, "bg": None} for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def auto_resize(self) -> bool:
        """Syncs canvas dimensions with current Rich Console terminal size."""
        cw, ch = self.console.width, self.console.height
        if cw != self.width or ch != self.height:
            self.resize(cw, ch)
            return True
        return False

    def enter_alternate_screen(self) -> None:
        self.console.show_cursor(False)
        self.console.file.write("\033[?1049h\033[H\033[2J")
        self.console.file.flush()

        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def exit_alternate_screen(self) -> None:
        self.console.file.write("\033[0m\033[?1049l")
        self.console.show_cursor(True)
        self.console.file.flush()

        if termios is not None and hasattr(self, "_old_term") and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def clear(self) -> None:
        empty = {"char": " ", "fg": None, "bg": None}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(self, x: int, y: int, text: str, fg=None, bg=None, style=0) -> None:
        if y < 0 or y >= self.height:
            return
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                self.grid[y][cx].update({"char": char, "fg": fg, "bg": bg})

    def edit_region_colors(self, x: int, y: int, w: int, h: int, fg=None, bg=None) -> None:
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                if fg: self.grid[cy][cx]["fg"] = fg
                if bg: self.grid[cy][cx]["bg"] = bg

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

                # Cursor positioning
                app(f"\033[{y+1};{x+1}H")

                curr_style_data = (row[x]["fg"], row[x]["bg"])
                char_buffer = []

                # Run-length encoding sweep across identical styles
                while x < self.width and (row[x]["fg"], row[x]["bg"]) == curr_style_data and row[x] != front_row[x]:
                    char_buffer.append(row[x]["char"])
                    front_row[x] = row[x].copy()
                    x += 1

                text = "".join(char_buffer)
                style = self._get_style(*curr_style_data)

                if style:
                    app(style.render(text))
                else:
                    app(text)

        # Write directly to Console file stream to bypass print parsing overhead
        if buf:
            self.console.file.write("".join(buf))
            self.console.file.flush()
