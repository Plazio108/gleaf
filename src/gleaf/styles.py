"""Style flags bitmask definition."""
from enum import IntFlag


class Modifiers(IntFlag):
    """
    Curses-compatible style modifier bitmask.
    Matches standard NCurses attribute bit offsets (bits 16-24) to allow
    seamless interoperability with Python curses while extending bits 25-31
    for modern terminal escape sequences.
    """
    NORMAL = NONE = 0

    # Curses Standard Attribute Bit Positions (bits 16-24)
    STANDOUT = 1 << 16          # 65536   (curses.A_STANDOUT)  -> \033[7m
    UNDERLINE = 1 << 17         # 131072  (curses.A_UNDERLINE) -> \033[4m
    REVERSE = 1 << 18           # 262144  (curses.A_REVERSE)   -> \033[7m
    BLINK = 1 << 19             # 524288  (curses.A_BLINK)     -> \033[5m
    DIM = 1 << 20               # 1048576 (curses.A_DIM)       -> \033[2m
    BOLD = 1 << 21              # 2097152 (curses.A_BOLD)      -> \033[1m
    ALTCHARSET = 1 << 22        # 4194304 (curses.A_ALTCHARSET)
    HIDDEN = 1 << 23            # 8388608 (curses.A_INVIS)     -> \033[8m
    PROTECT = 1 << 24           # 16777216(curses.A_PROTECT)

    # Extended Modern Terminal Attributes (bits 25-31)
    ITALIC = 1 << 25            # 33554432  -> \033[3m
    STRIKETHROUGH = 1 << 26     # 67108864  -> \033[9m
    OVERLINE = 1 << 27          # 134217728 -> \033[53m
    DOUBLE_UNDERLINE = 1 << 28  # 268435456 -> \033[4:2m or \033[21m
    CURLY_UNDERLINE = 1 << 29   # 536870912 -> \033[4:3m
    DOTTED_UNDERLINE = 1 << 30  # 1073741824-> \033[4:4m
    DASHED_UNDERLINE = 1 << 31  # 2147483648-> \033[4:5m

    # Backward-compatibility legacy bit shifts (bits 0-7)
    _LEGACY_BOLD = 1 << 0
    _LEGACY_DIM = 1 << 1
    _LEGACY_UNDERLINE = 1 << 2
    _LEGACY_REVERSE = 1 << 3
    _LEGACY_ITALIC = 1 << 4
    _LEGACY_STRIKE = 1 << 5
    _LEGACY_OVERLINE = 1 << 6
    _LEGACY_BLINK = 1 << 7

Style = Modifiers
