"""Base Interface for Canvas Backends."""

import shutil
import sys
import textwrap

from ..caps import TerminalCaps

try:
    import termios
    import tty
except ImportError:
    termios = None

UNSET = object()  # Sentinel to distinguish between explicit None vs unpassed args


class BaseCanvas:
    def __init__(self, width: int | None = None, height: int | None = None):
        self.caps = TerminalCaps()
        w, h = shutil.get_terminal_size((80, 24))
        self.width = width or w
        self.height = height or h

        # Save the external shell state as early as possible
        self._shell_mode = None
        self._prog_mode = None
        self._save_shell_mode()

    # --- Abstract Cell Getters ---
    def get_char(self, x: int, y: int) -> str:
        raise NotImplementedError

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        raise NotImplementedError

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
        raise NotImplementedError

    def get_style(self, x: int, y: int) -> int:
        raise NotImplementedError

    def get_cell(
        self, x: int, y: int
    ) -> tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None, int]:
        """Returns (char, fg, bg, style) at (x, y)."""
        return (
            self.get_char(x, y),
            self.get_fg(x, y),
            self.get_bg(x, y),
            self.get_style(x, y),
        )

    # --- Default Region / Zone Getters (Fallback using Cell Getters) ---
    def get_region_chars(self, x: int, y: int, w: int, h: int) -> list[list[str]]:
        return [
            [self.get_char(cx, cy) for cx in range(x, x + w)] for cy in range(y, y + h)
        ]

    def get_region_fg(
        self, x: int, y: int, w: int, h: int
    ) -> list[list[tuple[int, int, int] | None]]:
        return [
            [self.get_fg(cx, cy) for cx in range(x, x + w)] for cy in range(y, y + h)
        ]

    def get_region_bg(
        self, x: int, y: int, w: int, h: int
    ) -> list[list[tuple[int, int, int] | None]]:
        return [
            [self.get_bg(cx, cy) for cx in range(x, x + w)] for cy in range(y, y + h)
        ]

    def get_region_styles(self, x: int, y: int, w: int, h: int) -> list[list[int]]:
        return [
            [self.get_style(cx, cy) for cx in range(x, x + w)] for cy in range(y, y + h)
        ]

    def get_region_cells(
        self, x: int, y: int, w: int, h: int
    ) -> list[
        list[tuple[str, tuple[int, int, int] | None, tuple[int, int, int] | None, int]]
    ]:
        return [
            [self.get_cell(cx, cy) for cx in range(x, x + w)] for cy in range(y, y + h)
        ]

    # --- Canvas management ---
    def resize(self, width: int, height: int):
        raise NotImplementedError

    def auto_resize(self):
        w, h = shutil.get_terminal_size()
        if w != self.width or h != self.height:
            self.resize(w, h)

    def clear(self):
        raise NotImplementedError

    def put_str(
        self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=None
    ):
        """Draws string at (x,y). Unpassed parameters preserve existing cell properties."""
        raise NotImplementedError

    def put_block(
        self,
        x: int,
        y: int,
        text: str,
        max_w: int | None = None,
        max_h: int | None = None,
        fg=UNSET,
        bg=UNSET,
        style=UNSET,
        ul_fg=None,
        wrap: bool = True,
    ) -> int:
        """
        Prints multiline text block with optional word wrapping and height bounds.
        Returns the number of lines written.
        """
        lines = text.splitlines()
        formatted_lines: list[str] = []

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
                self.put_str(x, py, line_text, fg=fg, bg=bg, style=style, ul_fg=ul_fg)

        return len(lines_to_draw)

    def edit_region_colors(
        self, x: int, y: int, w: int, h: int, fg=UNSET, bg=UNSET, ul_fg=None
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
                    if ul_fg is not None:
                        cell.ul_fg = ul_fg

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

    def render(self, compute_only=False):
        raise NotImplementedError

    # def enter_alternate_screen(self):
    #     # \033[?25l hides the cursor
    #     sys.__stdout__.write("\033[?1049h\033[H\033[2J\033[?25l")
    #     sys.__stdout__.flush()

    #     # Disable terminal keystroke echoing
    #     if termios is not None:
    #         self._old_term = termios.tcgetattr(sys.stdin)
    #         tty.setcbreak(sys.stdin.fileno())

    # def exit_alternate_screen(self):
    #     # \033[?25h restores the cursor
    #     sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
    #     sys.__stdout__.flush()

    #     # Restore terminal keystroke echoing
    #     if termios is not None and hasattr(self, '_old_term'):
    #         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def _save_shell_mode(self):
        """Captures outer terminal state before gleaf modifies anything."""
        if termios is not None and self._shell_mode is None:
            try:
                self._shell_mode = termios.tcgetattr(sys.stdin.fileno())
            except (termios.error, ValueError, AttributeError):
                pass

    def enter_alternate_screen(self):
        # Guarantee shell mode was saved before entering prog mode
        self._save_shell_mode()

        sys.__stdout__.write("\033[0m\033[?1049h\033[H\033[2J\033[?25l")
        sys.__stdout__.flush()

        if termios is not None:
            tty.setcbreak(sys.stdin.fileno())
            # Capture the program/TUI mode state
            try:
                self._prog_mode = termios.tcgetattr(sys.stdin.fileno())
            except (termios.error, ValueError, AttributeError):
                pass

    def exit_alternate_screen(self):
        sys.__stdout__.write("\033[0m\033[?1049l\033[?25h")
        sys.__stdout__.flush()

        # Restore exact outer shell attributes recorded at startup
        if termios is not None and self._shell_mode is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSAFLUSH, self._shell_mode
                )
            except (termios.error, ValueError, AttributeError):
                pass

    def render_ansi_sequence(
        self, sequence: str | bytes, width: int, height: int
    ) -> str:
        """Parses an ANSI escape sequence (cursor positioning, newlines, text)
        and reconstructs it into a plain text string bounded within a width x height rectangle.

        Edge cases handled:
        - 1-indexed to 0-indexed terminal coordinate translation (\033[y;xH).
        - Out-of-bounds clamping and automatic line wrapping.
        - Decodes bytes streams automatically.
        - Malformed ANSI escape sequences.
        """
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8", errors="ignore")

        # Initialize a grid filled with spaces
        grid = [[" " for _ in range(width)] for _ in range(height)]

        cx = 0
        cy = 0
        i = 0
        n = len(sequence)

        while i < n:
            char = sequence[i]

            if char == "\033":
                # Check for CSI escape sequence
                if i + 1 < n and sequence[i + 1] == "[":
                    i += 2
                    param_str = ""
                    while i < n and not (sequence[i].isalpha() or sequence[i] == "~"):
                        param_str += sequence[i]
                        i += 1

                    if i < n:
                        command = sequence[i]
                        i += 1

                        # Cursor Position (CUP): \033[y;xH or \033[y;xf
                        if command in ("H", "f"):
                            parts = param_str.split(";")
                            if len(parts) >= 2:
                                try:
                                    # Terminal coordinates are 1-based; convert to 0-based
                                    cy = int(parts[0]) - 1 if parts[0] else 0
                                    cx = int(parts[1]) - 1 if parts[1] else 0
                                except ValueError:
                                    pass
                            elif len(parts) == 1 and parts[0] == "":
                                cy, cx = 0, 0
                else:
                    i += 1
            elif char == "\n":
                cy += 1
                cx = 0
                i += 1
            elif char == "\r":
                cx = 0
                i += 1
            else:
                # Printable character: write if within target rectangle bounds
                if 0 <= cy < height and 0 <= cx < width:
                    grid[cy][cx] = char

                cx += 1
                # Auto-wrap to next row if exceeding width
                if cx >= width:
                    cx = 0
                    cy += 1
                i += 1

        return "\n".join("".join(row) for row in grid)
