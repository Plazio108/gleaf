"""Terminal capability detection and adaptive color fallback engine."""

import os
from typing import Optional, Tuple, Union
from .styles import Style

RGB = Tuple[int, int, int]


def rgb_to_256(r: int, g: int, b: int) -> int:
    """Converts 24-bit RGB to the closest 256-color ANSI palette index."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round(((r - 8) / 247) * 23) + 232
    return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)


class TerminalCaps:
    """Detects terminal rendering capabilities and adapts escape sequences."""

    def __init__(self):
        colorterm = os.environ.get("COLORTERM", "").lower()
        term = os.environ.get("TERM", "").lower()
        no_color = "NO_COLOR" in os.environ

        self.no_color = no_color
        self.has_truecolor = not no_color and (
            colorterm in ("truecolor", "24bit")
            or "direct" in term
            or os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm", "kitty", "Alacritty")
            or "VSCODE_GIT_IPC_HANDLE" in os.environ
        )
        self.has_256color = not no_color and (
            self.has_truecolor or "256color" in term or "256" in term
        )
        self.has_modifiers = not os.environ.get("ANSI_COLORS_DISABLED")
        self.has_italic = self.has_modifiers and ("xterm" in term or "vt100" in term or self.has_truecolor)

    def filter_style(self, style: int) -> int:
        """Strips unsupported style modifiers based on terminal capabilities."""
        if not self.has_modifiers:
            return Style.NONE
        if not self.has_italic and (style & Style.ITALIC):
            style &= ~Style.ITALIC
        return style

    def format_style_ansi(
        self,
        fg: Optional[Union[RGB, int]] = None,
        bg: Optional[Union[RGB, int]] = None,
        style: int = Style.NONE,
    ) -> str:
        """Generates appropriate ANSI codes respecting color degradation settings."""
        if self.no_color:
            fg = None
            bg = None

        style = self.filter_style(style)
        codes = []

        if style:
            if style & Style.BOLD: codes.append("1")
            if style & Style.DIM: codes.append("2")
            if style & Style.ITALIC: codes.append("3")
            if style & Style.UNDERLINE: codes.append("4")
            if style & Style.BLINK: codes.append("5")
            if style & Style.REVERSE: codes.append("7")
            if style & Style.STRIKE: codes.append("9")

        # Foreground
        if fg is not None:
            if isinstance(fg, tuple) and len(fg) == 3:
                if self.has_truecolor:
                    codes.append(f"38;2;{fg[0]};{fg[1]};{fg[2]}")
                elif self.has_256color:
                    codes.append(f"38;5;{rgb_to_256(*fg)}")
            elif isinstance(fg, int):
                codes.append(f"38;5;{fg}")

        # Background
        if bg is not None:
            if isinstance(bg, tuple) and len(bg) == 3:
                if self.has_truecolor:
                    codes.append(f"48;2;{bg[0]};{bg[1]};{bg[2]}")
                elif self.has_256color:
                    codes.append(f"48;5;{rgb_to_256(*bg)}")
            elif isinstance(bg, int):
                codes.append(f"48;5;{bg}")

        return f"\x1b[{';'.join(codes)}m" if codes else ""
