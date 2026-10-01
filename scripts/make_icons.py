"""Draw the xPoints app icons (a white "xP" on the site's green) as PNG files.

No image libraries needed: shapes are drawn at 4x size and averaged down for smooth edges.
    python3 scripts/make_icons.py
Writes frontend/icons/: icon-512.png, icon-192.png, apple-touch-icon.png (180) and favicon-32.png.
Icons are full-bleed squares (iOS and Android round the corners themselves), and the mark sits
well inside the "maskable" safe zone.
"""
import math
import os
import struct
import zlib

GREEN = (31, 122, 77)
WHITE = (255, 255, 255)
SUPERSAMPLE = 4
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "icons")


def segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def arc_distance(px, py, cx, cy, r):
    """Distance to the right-hand half of a circle (the bowl of the P)."""
    if px < cx:
        return float("inf")
    return abs(math.hypot(px - cx, py - cy) - r)


def coverage(u, v):
    """Whether a point (u, v from 0 to 1) is on the "xP" mark. It sits well inside the maskable
    safe zone. A lone "x" looked like a close button, so the P makes it read as a name."""
    width = 0.078  # stroke thickness
    strokes = [
        (0.25, 0.47, 0.43, 0.69), (0.25, 0.69, 0.43, 0.47),   # x (lowercase height)
        (0.55, 0.31, 0.55, 0.69),                             # P stem
        (0.55, 0.31, 0.63, 0.31), (0.55, 0.53, 0.63, 0.53),   # P bowl, top and bottom
    ]
    if any(segment_distance(u, v, *seg) <= width / 2 for seg in strokes):
        return True
    return arc_distance(u, v, 0.63, 0.42, 0.11) <= width / 2


def render(size):
    big = size * SUPERSAMPLE
    rows = []
    for y in range(size):
        row = bytearray([0])  # PNG filter byte
        for x in range(size):
            hits = 0
            for sy in range(SUPERSAMPLE):
                for sx in range(SUPERSAMPLE):
                    u = (x * SUPERSAMPLE + sx + 0.5) / big
                    v = (y * SUPERSAMPLE + sy + 0.5) / big
                    hits += coverage(u, v)
            a = hits / SUPERSAMPLE ** 2
            row += bytes(round(GREEN[i] * (1 - a) + WHITE[i] * a) for i in range(3))
        rows.append(bytes(row))
    return png(size, size, b"".join(rows))


def png(width, height, raw):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, size in [("icon-512.png", 512), ("icon-192.png", 192), ("apple-touch-icon.png", 180), ("favicon-32.png", 32)]:
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(render(size))
        print(f"wrote {name} ({size}x{size})")


if __name__ == "__main__":
    main()
