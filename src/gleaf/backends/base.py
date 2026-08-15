"""Base Interface for Canvas Backends."""

import re
import shutil
import sys
import textwrap
import unicodedata

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

    # def render_ansi_sequence(
    #     self, sequence: str | bytes, width: int, height: int
    # ) -> str:
    #     """Parses a raw ANSI sequence, projects it to a strict width/height grid,
    #     and reconstructs an optimized, leak-proof ANSI string.
    #     """
    #     if isinstance(sequence, bytes):
    #         sequence = sequence.decode("utf-8", errors="ignore")

    #     # Virtual grids
    #     grid_char = [[" " for _ in range(width)] for _ in range(height)]
    #     grid_sgr = [["0" for _ in range(width)] for _ in range(height)]

    #     # ANSI State Machine Accumulator
    #     sgr_state = {"fg": None, "bg": None, "ul": None, "styles": set()}
    #     current_sgr = "0"
    #     cy, cx = 0, 0

    #     def parse_sgr(param_str: str):
    #         """Additively parses SGR strings to track exact formatting state."""
    #         parts = param_str.split(";") if param_str else ["0"]
    #         i = 0
    #         while i < len(parts):
    #             p = parts[i]
    #             code = int(p) if p else 0

    #             if code == 0:
    #                 sgr_state["fg"] = None
    #                 sgr_state["bg"] = None
    #                 sgr_state["ul"] = None
    #                 sgr_state["styles"].clear()
    #             elif code in (1, 2, 3, 4, 5, 7, 8, 9):
    #                 sgr_state["styles"].add(code)
    #             elif code == 22:
    #                 sgr_state["styles"].discard(1)
    #                 sgr_state["styles"].discard(2)
    #             elif code in (23, 24, 25, 27, 28, 29):
    #                 sgr_state["styles"].discard(
    #                     {23: 3, 24: 4, 25: 5, 27: 7, 28: 8, 29: 9}[code]
    #                 )
    #             elif 30 <= code <= 37 or 90 <= code <= 97:
    #                 sgr_state["fg"] = str(code)
    #             elif code == 39:
    #                 sgr_state["fg"] = None
    #             elif 40 <= code <= 47 or 100 <= code <= 107:
    #                 sgr_state["bg"] = str(code)
    #             elif code == 49:
    #                 sgr_state["bg"] = None
    #             elif code in (38, 48, 58):
    #                 if i + 2 < len(parts) and parts[i + 1] == "5":
    #                     val = f"{code};5;{parts[i + 2]}"
    #                     if code == 38:
    #                         sgr_state["fg"] = val
    #                     elif code == 48:
    #                         sgr_state["bg"] = val
    #                     elif code == 58:
    #                         sgr_state["ul"] = val
    #                     i += 2
    #                 elif i + 4 < len(parts) and parts[i + 1] == "2":
    #                     val = f"{code};2;{parts[i + 2]};{parts[i + 3]};{parts[i + 4]}"
    #                     if code == 38:
    #                         sgr_state["fg"] = val
    #                     elif code == 48:
    #                         sgr_state["bg"] = val
    #                     elif code == 58:
    #                         sgr_state["ul"] = val
    #                     i += 4
    #             elif code == 59:
    #                 sgr_state["ul"] = None
    #             i += 1

    #     def state_to_sgr() -> str:
    #         """Serializes current state to an absolute, reset-first SGR string."""
    #         codes = []
    #         if sgr_state["styles"]:
    #             codes.extend(str(s) for s in sorted(sgr_state["styles"]))
    #         if sgr_state["fg"]:
    #             codes.append(sgr_state["fg"])
    #         if sgr_state["bg"]:
    #             codes.append(sgr_state["bg"])
    #         if sgr_state["ul"]:
    #             codes.append(sgr_state["ul"])
    #         return "0;" + ";".join(codes) if codes else "0"

    #     i = 0
    #     n = len(sequence)

    #     while i < n:
    #         char = sequence[i]

    #         if char == "\033":
    #             if i + 1 < n and sequence[i + 1] == "[":
    #                 i += 2
    #                 param_str = ""
    #                 while i < n and not (sequence[i].isalpha() or sequence[i] == "~"):
    #                     param_str += sequence[i]
    #                     i += 1

    #                 if i < n:
    #                     command = sequence[i]
    #                     i += 1

    #                     if command in ("H", "f"):
    #                         parts = param_str.split(";")
    #                         if len(parts) >= 2:
    #                             try:
    #                                 cy = int(parts[0]) - 1 if parts[0] else 0
    #                                 cx = int(parts[1]) - 1 if parts[1] else 0
    #                             except ValueError:
    #                                 pass
    #                         elif len(parts) == 1 and parts[0] == "":
    #                             cy, cx = 0, 0
    #                     elif command == "m":
    #                         parse_sgr(param_str)
    #                         current_sgr = state_to_sgr()
    #                     elif command == "K":  # Erase in line
    #                         p = int(param_str) if param_str else 0
    #                         start_c = cx if p == 0 else 0
    #                         end_c = width if p in (0, 2) else cx + 1
    #                         for c in range(max(0, start_c), min(width, end_c)):
    #                             if 0 <= cy < height:
    #                                 grid_char[cy][c] = " "
    #                                 grid_sgr[cy][c] = current_sgr
    #                     elif command == "J":  # Clear screen
    #                         if param_str == "2":
    #                             for r in range(height):
    #                                 for c in range(width):
    #                                     grid_char[r][c] = " "
    #                                     grid_sgr[r][c] = current_sgr
    #             else:
    #                 i += 1
    #         elif char == "\n":
    #             cy += 1
    #             cx = 0
    #             i += 1
    #         elif char == "\r":
    #             cx = 0
    #             i += 1
    #         elif char == "\b":
    #             cx = max(0, cx - 1)
    #             i += 1
    #         elif char == "\t":
    #             cx = (cx + 8) & ~7
    #             i += 1
    #         else:
    #             # Handle wide characters (e.g., CJK) taking 2 columns
    #             w = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1

    #             if 0 <= cy < height and 0 <= cx < width:
    #                 grid_char[cy][cx] = char
    #                 grid_sgr[cy][cx] = current_sgr
    #                 # Reserve the extra cell so layouts don't misalign
    #                 if w == 2 and cx + 1 < width:
    #                     grid_char[cy][cx + 1] = ""
    #                     grid_sgr[cy][cx + 1] = current_sgr

    #             cx += w
    #             if cx >= width:
    #                 cx = 0
    #                 cy += 1
    #             i += 1

    #     # Serialize back into strings
    #     output_lines = []
    #     for r in range(height):
    #         # Right-strip purely empty cells to prevent terminal wrap-around SGR bleeding
    #         max_c = width - 1
    #         while max_c >= 0:
    #             if grid_char[r][max_c] != " " or grid_sgr[r][max_c] != "0":
    #                 break
    #             max_c -= 1

    #         line_parts = []
    #         last_sgr = None
    #         for c in range(max_c + 1):
    #             ch = grid_char[r][c]
    #             sgr = grid_sgr[r][c]

    #             if sgr != last_sgr:
    #                 line_parts.append(f"\033[{sgr}m")
    #                 last_sgr = sgr
    #             line_parts.append(ch)

    #         # Hard reset format before the line ends to guarantee safety
    #         if last_sgr is not None and last_sgr != "0":
    #             line_parts.append("\033[0m")

    #         output_lines.append("".join(line_parts))

    #     return "\n".join(output_lines)

    def render_ansi_sequence(self, ansi_seq: str, width: int, height: int) -> str:
        """
        Parses a raw ANSI escape string (with cursor movements and SGR styles),
        and reconstructs it as a printable multi-line string cropped to the given bounds.
        """
        # Regex to match CSI sequences (e.g., \033[1;31m or \033[10;5H)
        ansi_regex = re.compile(r"\033\[([0-9;]*)([a-zA-Z])")

        # Virtual grid to hold characters and their active SGR style sequences
        chars = [[" " for _ in range(width)] for _ in range(height)]
        styles = [["" for _ in range(width)] for _ in range(height)]

        cx, cy = 0, 0
        current_style = ""

        # 1. Parse the input string and plot it onto the virtual grid
        pos = 0
        for match in ansi_regex.finditer(ansi_seq):
            # Dump any literal text before this escape code to the grid
            text = ansi_seq[pos : match.start()]
            for char in text:
                if char == "\n":
                    cy += 1
                    continue
                if char == "\r":
                    cx = 0
                    continue

                if 0 <= cy < height and 0 <= cx < width:
                    chars[cy][cx] = char
                    styles[cy][cx] = current_style
                cx += 1

            # Process the ANSI escape code
            params, command = match.groups()

            if command in ("H", "f"):  # Absolute cursor positioning
                parts = params.split(";")
                r = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 1
                c = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                cy, cx = r - 1, c - 1

            elif command == "m":  # SGR color/style modification
                if params in ("", "0"):
                    current_style = ""
                elif params.startswith("0;"):
                    # Hard reset + new style overrides everything
                    current_style = f"\033[{params}m"
                else:
                    # Stack additional styles (e.g. adding bold on top of red)
                    current_style += f"\033[{params}m"

            pos = match.end()

        # Dump any trailing text after the final escape sequence
        text = ansi_seq[pos:]
        for char in text:
            if char == "\n":
                cy += 1
                continue
            if char == "\r":
                cx = 0
                continue
            if 0 <= cy < height and 0 <= cx < width:
                chars[cy][cx] = char
                styles[cy][cx] = current_style
            cx += 1

        # 2. Reconstruct the final string with strict style isolation
        output_lines = []
        for y in range(height):
            line_buf = []
            active_style = ""

            for x in range(width):
                cell_style = styles[y][x]

                if cell_style != active_style:
                    if cell_style == "":
                        line_buf.append("\033[0m")
                    else:
                        # Foolproof state transfer: wipe the slate clean, then apply
                        # the cell's accumulated style. This guarantees previous
                        # cell attributes (like backgrounds) don't bleed into this one.
                        line_buf.append(f"\033[0m{cell_style}")
                    active_style = cell_style

                line_buf.append(chars[y][x])

            # Cap every line with a hard reset if a style was active.
            # This solves the cropped line-wrap edge case, preventing bleed.
            if active_style != "":
                line_buf.append("\033[0m")

            output_lines.append("".join(line_buf))

        # Join with standard newlines (no need for cursor resets to print this normally)
        return "\n".join(output_lines)
