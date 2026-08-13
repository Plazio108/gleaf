"""Vanilla Python Terminal Alternate Screen FPS Benchmark."""

import sys
import time
import shutil

def run_benchmark(duration: float = 5.0):
    # Enter alternate screen and hide cursor
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()

    size = shutil.get_terminal_size()
    w, h = size.columns, size.lines

    frame_count = 0
    start_time = time.time()
    
    chars = [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*#@_+-="]
    num_chars = len(chars)

    try:
        while time.time() - start_time < duration:
            buf = bytearray()
            # Position cursor at top left once per frame
            buf.extend(b"\033[H")
            
            for y in range(h):
                buf.extend(f"\033[{y+1};1H".encode('ascii'))
                for x in range(w):
                    c = chars[(x + y + frame_count) % num_chars]
                    buf.append(c)
            
            sys.stdout.buffer.write(buf)
            sys.stdout.flush()
            frame_count += 1

    finally:
        # Restore normal screen and show cursor
        sys.stdout.write("\033[?1049l\033[?25h")
        sys.stdout.flush()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    print(f"\nRendered {frame_count} frames in {elapsed:.2f} seconds.")
    print(f"Max Raw Terminal FPS: {fps:.2f}\n")

if __name__ == "__main__":
    run_benchmark()
