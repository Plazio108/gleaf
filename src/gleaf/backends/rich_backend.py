"""High-Performance Rich Backend with Single-Pass Frame Buffering."""

from rich.console import Console
from rich.style import Style

try:
    from ..caps import Modifiers, TerminalCaps
except ImportError:
    from gleaf.caps import Modifiers, TerminalCaps

from .base import UNSET, BaseCanvas


class RichCanvas(BaseCanvas):
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        console: Console | None = None,
    ):
        self.console = console or Console()

        # Default to Console dimensions if not explicitly passed
        w = width if width is not None else self.console.width
        h = height if height is not None else self.console.height

        super().__init__(w, h)
        self.caps = TerminalCaps()
        self._style_cache = {}

        self.grid = self._create_grid(self.width, self.height)
        self.front_buffer = self._create_grid(self.width, self.height)

    def _create_grid(self, w: int, h: int):
        return [
            [
                {"char": " ", "fg": None, "bg": None, "style": 0, "ul_fg": None}
                for _ in range(w)
            ]
            for _ in range(h)
        ]

    def _get_style(
        self,
        fg: tuple[int, int, int] | None,
        bg: tuple[int, int, int] | None,
        style_flags: int,
        ul_fg: tuple[int, int, int] | None = None,
    ) -> Style | None:
        key = (fg, bg, style_flags, ul_fg)
        if key in self._style_cache:
            return self._style_cache[key]

        color = f"rgb({fg[0]},{fg[1]},{fg[2]})" if fg else None
        bgcolor = f"rgb({bg[0]},{bg[1]},{bg[2]})" if bg else None

        # Map Modifiers bitmask
        bold = bool(style_flags & Modifiers.BOLD)
        dim = bool(style_flags & Modifiers.DIM)
        italic = bool(style_flags & Modifiers.ITALIC)

        ext_ul = bool(
            style_flags
            & (
                Modifiers.CURLY_UNDERLINE
                | Modifiers.DOTTED_UNDERLINE
                | Modifiers.DASHED_UNDERLINE
            )
        )
        underline = bool(style_flags & Modifiers.UNDERLINE) or ext_ul
        underline2 = bool(style_flags & Modifiers.DOUBLE_UNDERLINE)

        blink = bool(style_flags & Modifiers.BLINK)
        reverse = bool(style_flags & (Modifiers.REVERSE | Modifiers.STANDOUT))
        strike = bool(style_flags & Modifiers.STRIKETHROUGH)
        overline = bool(style_flags & Modifiers.OVERLINE)
        conceal = bool(style_flags & Modifiers.HIDDEN)

        kwargs = {}
        if color:
            kwargs["color"] = color
        if bgcolor:
            kwargs["bgcolor"] = bgcolor
        if bold:
            kwargs["bold"] = True
        if dim:
            kwargs["dim"] = True
        if italic:
            kwargs["italic"] = True
        if underline:
            kwargs["underline"] = True
        if underline2:
            kwargs["underline2"] = True
        if blink:
            kwargs["blink"] = True
        if reverse:
            kwargs["reverse"] = True
        if strike:
            kwargs["strike"] = True
        if overline:
            kwargs["overline"] = True
        if conceal:
            kwargs["conceal"] = True

        try:
            style = Style(**kwargs) if kwargs else None
        except Exception:
            # Fallback for versions of Rich that may not accept overline or underline2
            kwargs.pop("overline", None)
            kwargs.pop("underline2", None)
            if underline2:
                kwargs["underline"] = True
            style = Style(**kwargs) if kwargs else None

        self._style_cache[key] = style
        return style

    # --- Cell Inspection Implementations ---
    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["char"]
        return " "

    def get_fg(self, x: int, y: int) -> tuple[int, int, int] | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]["fg"]
        return None

    def get_bg(self, x: int, y: int) -> tuple[int, int, int] | None:
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
            [
                {"char": None, "fg": None, "bg": None, "style": None, "ul_fg": None}
                for _ in range(self.width)
            ]
            for _ in range(self.height)
        ]

    def auto_resize(self) -> bool:
        cw, ch = self.console.width, self.console.height
        if cw != self.width or ch != self.height:
            self.resize(cw, ch)
            return True
        return False

    def enter_alternate_screen(self) -> None:
        super().enter_alternate_screen()
        self.console.set_alt_screen(True)
        self.console.show_cursor(False)

    def exit_alternate_screen(self) -> None:
        super().exit_alternate_screen()
        self.console.set_alt_screen(False)
        self.console.show_cursor(True)

    def clear(self) -> None:
        empty = {"char": " ", "fg": None, "bg": None, "style": 0, "ul_fg": None}
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = empty.copy()

    def put_str(
        self, x: int, y: int, text: str, fg=UNSET, bg=UNSET, style=UNSET, ul_fg=UNSET
    ) -> None:
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
                if ul_fg is not UNSET:
                    cell["ul_fg"] = ul_fg

    def edit_region_colors(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        fg=UNSET,
        bg=UNSET,
        style=UNSET,
        ul_fg=UNSET,
    ) -> None:
        for cy in range(max(0, y), min(self.height, y + h)):
            for cx in range(max(0, x), min(self.width, x + w)):
                cell = self.grid[cy][cx]
                if fg is not UNSET:
                    cell["fg"] = fg
                if bg is not UNSET:
                    cell["bg"] = bg
                if style is not UNSET:
                    cell["style"] = style
                if ul_fg is not UNSET:
                    cell["ul_fg"] = ul_fg

    def render(self, compute_only=False):
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

                app(f"\033[{y + 1};{x + 1}H")

                curr_style_data = (
                    row[x]["fg"],
                    row[x]["bg"],
                    row[x]["style"],
                    row[x]["ul_fg"],
                )
                char_buffer = []

                while (
                    x < self.width
                    and (row[x]["fg"], row[x]["bg"], row[x]["style"], row[x]["ul_fg"])
                    == curr_style_data
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

        if compute_only:
            return "".join(buf)

        if buf:
            self.console.file.write("".join(buf))
            self.console.file.flush()
