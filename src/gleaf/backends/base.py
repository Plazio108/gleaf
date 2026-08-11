"""Base Interface for Canvas Backends."""

import sys
import shutil
import textwrap
from typing import Optional, List
from ..caps import TerminalCaps
from ..styles import Style

import sys
try:
    import termios
    import tty
except ImportError:
    termios = None

UNSET = object()  # Sentinel to distinguish between explicit None vs unpassed args


class BaseCanvas:
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None):
        self.caps = TerminalCaps()
        w, h = shutil.get_terminal_size((80, 24))
        self.width = width or w
        self.height = height or h

    def resize(self, width: int, height: int):
        raise NotImplementedError

    def auto_resize(self):
        w, h = shutil.get_terminal_size()
        if w != self.width or h != self.height:
            self.resize(w, h)

    def clear(self):
        raise NotImplementedError

    def put_str(self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET):
        """Draws string at (x,y). Unpassed parameters preserve existing cell properties."""
        raise NotImplementedError

    def put_block(
        self,
        x: int,
        y: int,
        text: str,
        max_w: Optional[int] = None,
        max_h: Optional[int] = None,
        fg=UNSET,
        bg=UNSET,
        style=UNSET,
        wrap: bool = True,
    ) -> int:
        """
        Prints multiline text block with optional word wrapping and height bounds.
        Returns the number of lines written.
        """
        lines = text.splitlines()
        formatted_lines: List[str] = []

        eff_w = max_w if max_w is not None else (self.width - x)
        if eff_w <= 0:
            return 0

        for line in lines:
            if wrap and len(line) > eff_w:
                wrapped = textwrap.wrap(line, width=eff_w)
                formatted_lines.extend(wrapped if wrapped else [""])
            else:
                formatted_lines.append(line[:eff_w])

        lines_to_draw = formatted_lines
        if max_h is not None:
            lines_to_draw = lines_to_draw[:max_h]

        for i, line_text in enumerate(lines_to_draw):
            py = y + i
            if py >= self.height:
                break
            if py >= 0:
                self.put_str(x, py, line_text, fg=fg, bg=bg, style=style)

        return len(lines_to_draw)

    def edit_region_colors(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        fg=UNSET,
        bg=UNSET,
    ):
        """Edits foreground and/or background colors in a bounding box without affecting text or styles."""
        x1, x2 = max(0, x), min(self.width, x + w)
        y1, y2 = max(0, y), min(self.height, y + h)

        if hasattr(self, "_buffer"):
            for py in range(y1, y2):
                for px in range(x1, x2):
                    cell = self._buffer[py][px]
                    if fg is not UNSET:
                        cell.fg = fg
                    if bg is not UNSET:
                        cell.bg = bg

    def edit_region_style(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        style: int,
        mode: str = "add",
    ):
        """
        Edits style flags in a bounding box without modifying text or colors.
        mode options: 'add', 'remove', 'set', 'toggle'
        """
        x1, x2 = max(0, x), min(self.width, x + w)
        y1, y2 = max(0, y), min(self.height, y + h)

        if hasattr(self, "_buffer"):
            for py in range(y1, y2):
                for px in range(x1, x2):
                    cell = self._buffer[py][px]
                    if mode in ("add", "enable"):
                        cell.style |= style
                    elif mode in ("remove", "disable"):
                        cell.style &= ~style
                    elif mode in ("set", "replace"):
                        cell.style = style
                    elif mode == "toggle":
                        cell.style ^= style

    def render(self):
        raise NotImplementedError

    def enter_alternate_screen(self):
        # \033[?25l hides the cursor
        sys.__stdout__.write("\033[?1049h\033[H\033[2J\033[?25l")
        sys.__stdout__.flush()
        
        # Disable terminal keystroke echoing
        if termios is not None:
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            
    def exit_alternate_screen(self):
        # \033[?25h restores the cursor
        sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
        sys.__stdout__.flush()
        
        # Restore terminal keystroke echoing
        if termios is not None and hasattr(self, '_old_term'):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
