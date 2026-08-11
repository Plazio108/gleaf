"""Style flags bitmask definition."""

class Style:
    NONE = 0
    BOLD = 1 << 0
    DIM = 1 << 1
    ITALIC = 1 << 2
    UNDERLINE = 1 << 3
    BLINK = 1 << 4
    REVERSE = 1 << 5
    STRIKE = 1 << 6
