"""Terminal capability detection and adaptive color/style fallback engine."""

import os
from typing import Optional, Tuple, Union
from .styles import Modifiers

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

    def __init__(
        self,
        has_truecolor: Optional[bool] = None,
        has_256color: Optional[bool] = None,
        has_extended_underline: Optional[bool] = None,
        has_modifiers: Optional[bool] = None,
    ):
        colorterm = os.environ.get("COLORTERM", "").lower()
        term = os.environ.get("TERM", "").lower()
        term_program = os.environ.get("TERM_PROGRAM", "")
        no_color = "NO_COLOR" in os.environ

        try:
            vte_version = int(os.environ.get("VTE_VERSION", 0))
        except ValueError:
            vte_version = 0

        self.no_color = no_color

        # Color support detection with optional override support
        detected_truecolor = not no_color and (
            colorterm in ("truecolor", "24bit")
            or "direct" in term
            or term_program in ("iTerm.app", "WezTerm", "kitty", "Alacritty", "Ghostty")
            or "KITTY_WINDOW_ID" in os.environ
            or "VSCODE_GIT_IPC_HANDLE" in os.environ
        )
        self.has_truecolor = has_truecolor if has_truecolor is not None else detected_truecolor

        detected_256color = not no_color and (
            self.has_truecolor or "256color" in term or "256" in term
        )
        self.has_256color = has_256color if has_256color is not None else detected_256color

        # Modifier capability detection
        detected_modifiers = not os.environ.get("ANSI_COLORS_DISABLED")
        self.has_modifiers = has_modifiers if has_modifiers is not None else detected_modifiers

        self.has_italic = self.has_modifiers and (
            "xterm" in term or "vt100" in term or self.has_truecolor
        )
        self.has_overline = self.has_modifiers and (
            self.has_truecolor or "xterm" in term or "kitty" in term or "foot" in term
        )

        detected_ext_ul = self.has_modifiers and (
            "KITTY_WINDOW_ID" in os.environ
            or term_program in ("kitty", "WezTerm", "Ghostty", "iTerm.app")
            or any(t in term for t in ("kitty", "wezterm", "foot", "ghostty", "alacritty"))
            or vte_version >= 5000
        )
        self.has_extended_underline = (
            has_extended_underline if has_extended_underline is not None else detected_ext_ul
        )

    def filter_style(self, style: int) -> int:
        """Strips or degrades unsupported style modifiers."""
        if not self.has_modifiers:
            return Modifiers.NORMAL

        if not self.has_italic and (style & Modifiers.ITALIC):
            style &= ~Modifiers.ITALIC

        if not self.has_overline and (style & Modifiers.OVERLINE):
            style &= ~Modifiers.OVERLINE

        if not self.has_extended_underline:
            extended_underline_mask = (
                Modifiers.CURLY_UNDERLINE
                | Modifiers.DOTTED_UNDERLINE
                | Modifiers.DASHED_UNDERLINE
            )
            if style & extended_underline_mask:
                style &= ~extended_underline_mask
                style |= Modifiers.UNDERLINE

        return style

    def format_style_ansi(
        self,
        fg: Optional[Union[RGB, int]] = None,
        bg: Optional[Union[RGB, int]] = None,
        style: int = Modifiers.NORMAL,
        ul_fg: Optional[Union[RGB, int]] = None,
    ) -> str:
        if self.no_color:
            fg = None
            bg = None
            ul_fg = None

        style = self.filter_style(style)
        codes = []

        if style:
            # Bold
            if style & (Modifiers.BOLD | Modifiers._LEGACY_BOLD):
                codes.append("1")
            # Dim
            if style & (Modifiers.DIM | Modifiers._LEGACY_DIM):
                codes.append("2")
            # Italic
            if style & (Modifiers.ITALIC | Modifiers._LEGACY_ITALIC):
                codes.append("3")
            # Blink
            if style & (Modifiers.BLINK | Modifiers._LEGACY_BLINK):
                codes.append("5")
            # Reverse / Standout
            if style & (Modifiers.REVERSE | Modifiers.STANDOUT | Modifiers._LEGACY_REVERSE):
                codes.append("7")
            # Hidden / Conceal
            if style & Modifiers.HIDDEN:
                codes.append("8")
            # Strikethrough
            if style & (Modifiers.STRIKETHROUGH | Modifiers._LEGACY_STRIKE):
                codes.append("9")
            # Overline
            if style & (Modifiers.OVERLINE | Modifiers._LEGACY_OVERLINE):
                codes.append("53")

            # Underlines (Standard SGR 4 is used for default UNDERLINE to avoid mux bugs)
            if style & Modifiers.CURLY_UNDERLINE and self.has_extended_underline:
                codes.append("4:3")
            elif style & Modifiers.DOTTED_UNDERLINE and self.has_extended_underline:
                codes.append("4:4")
            elif style & Modifiers.DASHED_UNDERLINE and self.has_extended_underline:
                codes.append("4:5")
            elif style & Modifiers.DOUBLE_UNDERLINE:
                codes.append("4:2" if self.has_extended_underline else "21")
            elif style & (Modifiers.UNDERLINE | Modifiers._LEGACY_UNDERLINE):
                codes.append("4")

        # Foreground
        if fg is not None:
            if isinstance(fg, tuple) and len(fg) == 3:
                codes.append(f"38;2;{fg[0]};{fg[1]};{
                             fg[2]}" if self.has_truecolor else f"38;5;{rgb_to_256(*fg)}")
            elif isinstance(fg, int):
                codes.append(f"38;5;{fg}")

        # Background
        if bg is not None:
            if isinstance(bg, tuple) and len(bg) == 3:
                codes.append(f"48;2;{bg[0]};{bg[1]};{
                             bg[2]}" if self.has_truecolor else f"48;5;{rgb_to_256(*bg)}")
            elif isinstance(bg, int):
                codes.append(f"48;5;{bg}")

        # Underline Color (SGR 58)
        if ul_fg is not None and self.has_extended_underline:
            if isinstance(ul_fg, tuple) and len(ul_fg) == 3:
                codes.append(f"58;2;{ul_fg[0]};{ul_fg[1]};{
                             ul_fg[2]}" if self.has_truecolor else f"58;5;{rgb_to_256(*ul_fg)}")
            elif isinstance(ul_fg, int):
                codes.append(f"58;5;{ul_fg}")

        return f"\033[{';'.join(codes)}m" if codes else ""
