"""High-Performance Rich Backend with Single-Pass Frame Buffering."""

import sys
from rich.style import Style
from .base import BaseCanvas

try:
    import termios
    import tty
except ImportError:
    termios = None


class RichCanvas(BaseCanvas):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        self._style_cache = {}
        
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w, h):
        return [
            [{"char": " ", "fg": None, "bg": None} for _ in range(w)] 
            for _ in range(h)
        ]

    def _get_style(self, fg, bg):
        key = (fg, bg)
        if key in self._style_cache:
            return self._style_cache[key]
            
        color = f"rgb({fg[0]},{fg[1]},{fg[2]})" if fg else None
        bgcolor = f"rgb({bg[0]},{bg[1]},{bg[2]})" if bg else None
        
        style = Style(color=color, bgcolor=bgcolor) if (color or bgcolor) else None
        self._style_cache[key] = style
        return style

    def enter_alternate_screen(self):
        sys.stdout.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.stdout.flush()

        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def exit_alternate_screen(self):
        sys.stdout.write("\033[0m\033[?1049l\033[?25h")
        sys.stdout.flush()

        if termios is not None and hasattr(self, "_old_term") and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def clear(self):
        empty = {"char": " ", "fg": None, "bg": None}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(self, x, y, text, fg=None, bg=None, style=0):
        if y < 0 or y >= self.height: return
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                self.grid[y][cx].update({"char": char, "fg": fg, "bg": bg})

    def edit_region_colors(self, x, y, w, h, fg=None, bg=None):
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                if fg: self.grid[cy][cx]["fg"] = fg
                if bg: self.grid[cy][cx]["bg"] = bg

    def render(self):
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
                    
                # Position cursor only where changes occur
                app(f"\033[{y+1};{x+1}H")
                
                curr_style_data = (row[x]["fg"], row[x]["bg"])
                char_buffer = []
                
                while x < self.width and (row[x]["fg"], row[x]["bg"]) == curr_style_data and row[x] != front_row[x]:
                    char_buffer.append(row[x]["char"])
                    front_row[x] = row[x].copy()
                    x += 1
                    
                text = "".join(char_buffer)
                style = self._get_style(*curr_style_data)
                
                # Render using Rich's cached style directly to ANSI string without print() overhead
                if style:
                    app(style.render(text))
                else:
                    app(text)

        if buf:
            sys.stdout.write("".join(buf))
            sys.stdout.flush()
