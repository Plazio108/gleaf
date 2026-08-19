"""Core Texture structures, modes, and binary definitions."""
import struct

MODE_TRANSPARENT = 0
MODE_SET = 1
MODE_CLEAR = 2

# 24-byte struct layout:
# < I      : char (uint32)
#   3B B   : fg_r, fg_g, fg_b, fg_mode
#   3B B   : bg_r, bg_g, bg_b, bg_mode
#   3B B   : ul_r, ul_g, ul_b, ul_mode
#   I      : style (uint32)
#   B      : style_mode
#   3x     : padding (to 24 bytes)
CELL_STRUCT_FMT = "<I 3B B 3B B 3B B I B 3x"
CELL_BYTES = 24

# The empty tuple matched to iter_unpack (15 distinct extracted values)
EMPTY_CELL = (0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0)

HAS_NUMPY = False
TEXTURE_DTYPE = None
try:
    import numpy as np
    HAS_NUMPY = True
    TEXTURE_DTYPE = np.dtype([
        ('char', np.uint32),
        ('fg_r', np.uint8), ('fg_g', np.uint8), ('fg_b', np.uint8), ('fg_mode', np.uint8),
        ('bg_r', np.uint8), ('bg_g', np.uint8), ('bg_b', np.uint8), ('bg_mode', np.uint8),
        ('ul_r', np.uint8), ('ul_g', np.uint8), ('ul_b', np.uint8), ('ul_mode', np.uint8),
        ('style', np.uint32),
        ('style_mode', np.uint8),
        ('pad', np.uint8, 3)
    ])
except ImportError:
    pass
