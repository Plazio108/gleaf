"""Pure Python Double-Buffered Delta Renderer."""

import sys
import os
from .base import BaseCanvas

try:
    import termios
    import tty
except ImportError:
    termios = None


class PurePythonCanvas(BaseCanvas):
    def __init__(self, width=None, height=None):
        super().__init__(width, height)
        
        # Double Buffering
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)
        
        # Stateful ANSI Tracking
        self._current_fg = None
        self._current_bg = None
        self._current_style = 0
        self._cursor_x = -1
        self._cursor_y = -1

    def _create_grid(self, w, h):
        return [
            [{'char': ' ', 'fg': None, 'bg': None, 'style': 0} for _ in range(w)] 
            for _ in range(h)
        ]

    def _invalidate_front_buffer(self):
        for y in range(self.height):
            for x in range(self.width):
                self.front_buffer[y][x]['char'] = None 
        self._current_fg = None
        self._current_bg = None
        self._current_style = -1
        self._cursor_x = -1
        self._cursor_y = -1

    def resize(self, width: int, height: int):
        super().resize(width, height)
        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)
        sys.__stdout__.write("\033[2J")
        sys.__stdout__.flush()
        self._invalidate_front_buffer()

    def enter_alternate_screen(self):
        # \033[?1049h = alt screen, \033[H = home cursor, \033[2J = clear, \033[?25l = hide cursor
        sys.__stdout__.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.__stdout__.flush()
        
        # Disable echo and set cbreak mode
        if termios is not None and sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
        self._invalidate_front_buffer()

    def exit_alternate_screen(self):
        # \033[0m = reset styles, \033[?1049l = exit alt screen, \033[?25h = show cursor
        sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
        sys.__stdout__.flush()
        
        if termios is not None and hasattr(self, '_old_term') and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def clear(self):
        empty = {'char': ' ', 'fg': None, 'bg': None, 'style': 0}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(self, x, y, text, fg=None, bg=None, style=0):
        if y < 0 or y >= self.height:
            return
        
        for i, char in enumerate(text):
            cx = x + i
            if 0 <= cx < self.width:
                cell = self.grid[y][cx]
                cell['char'] = char
                if fg is not None: cell['fg'] = fg
                if bg is not None: cell['bg'] = bg
                if style is not None: cell['style'] = style

    def edit_region_colors(self, x, y, w, h, fg=None, bg=None):
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not None: cell['fg'] = fg
                if bg is not None: cell['bg'] = bg

    def render(self):
        out = []
        app = out.append
        
        grid = self.grid
        front = self.front_buffer
        
        cur_fg = self._current_fg
        cur_bg = self._current_bg
        cur_style = self._current_style
        cx = self._cursor_x
        cy = self._cursor_y
        
        for y in range(self.height):
            row_back = grid[y]
            row_front = front[y]
            
            for x in range(self.width):
                cell_back = row_back[x]
                cell_front = row_front[x]
                
                if cell_back == cell_front:
                    continue
                
                row_front[x] = cell_back.copy()
                
                if cx != x or cy != y:
                    app(f"\033[{y+1};{x+1}H")
                
                style = cell_back['style']
                fg = cell_back['fg']
                bg = cell_back['bg']
                
                style_changed = False
                if style != cur_style:
                    app("\033[0m")
                    cur_style = style
                    cur_fg = None
                    cur_bg = None
                    style_changed = True
                    
                    if style > 0:
                        if style & 1: app("\033[1m")
                        if style & 2: app("\033[2m")
                        if style & 4: app("\033[3m")
                        if style & 8: app("\033[4m")
                
                if fg != cur_fg or style_changed:
                    if fg is None:
                        app("\033[39m")
                    else:
                        app(f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m")
                    cur_fg = fg
                
                if bg != cur_bg or style_changed:
                    if bg is None:
                        app("\033[49m")
                    else:
                        app(f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m")
                    cur_bg = bg
                
                app(cell_back['char'])
                cx = x + 1
                cy = y

        self._current_fg = cur_fg
        self._current_bg = cur_bg
        self._current_style = cur_style
        self._cursor_x = cx
        self._cursor_y = cy
        
        if out:
            sys.__stdout__.write("".join(out))
            sys.__stdout__.flush()
