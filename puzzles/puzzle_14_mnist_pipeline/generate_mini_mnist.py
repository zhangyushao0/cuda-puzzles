#!/usr/bin/env python3
"""Generate a minimal 100-image MNIST subset in IDX format for testing.

Uses deterministic PRNG so the binary files are reproducible.
The generated images have realistic-looking pixel distributions
(mostly dark with some bright areas, like real handwritten digits).
"""

import struct
import random

NUM_IMAGES = 100
ROWS = 28
COLS = 28


def write_big_endian_uint32(f, val):
    """Write a 32-bit unsigned integer in big-endian format."""
    f.write(struct.pack(">I", val))


def generate_images(path, seed=42):
    """Generate IDX3-UBYTE image file with 100 synthetic digit-like images."""
    rng = random.Random(seed)

    with open(path, "wb") as f:
        # Header: magic=0x00000803, count, rows, cols (all big-endian)
        write_big_endian_uint32(f, 0x00000803)
        write_big_endian_uint32(f, NUM_IMAGES)
        write_big_endian_uint32(f, ROWS)
        write_big_endian_uint32(f, COLS)

        # Generate pixel data
        for _ in range(NUM_IMAGES):
            pixels = bytearray(ROWS * COLS)
            # Create digit-like pattern: mostly dark background with some strokes
            # Place 3-5 random bright strokes
            num_strokes = rng.randint(3, 5)
            for _ in range(num_strokes):
                # Random start position
                r = rng.randint(4, ROWS - 5)
                c = rng.randint(4, COLS - 5)
                length = rng.randint(3, 8)
                direction = rng.choice([(0, 1), (1, 0), (1, 1), (1, -1)])
                for step in range(length):
                    nr = r + step * direction[0]
                    nc = c + step * direction[1]
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        # Bright pixel with some variation
                        val = rng.randint(180, 255)
                        pixels[nr * COLS + nc] = val
                        # Anti-aliasing neighbors
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nnr, nnc = nr + dr, nc + dc
                            if 0 <= nnr < ROWS and 0 <= nnc < COLS:
                                existing = pixels[nnr * COLS + nnc]
                                new_val = rng.randint(60, 140)
                                pixels[nnr * COLS + nnc] = max(existing, new_val)

            # Add slight background noise
            for i in range(ROWS * COLS):
                if pixels[i] == 0 and rng.random() < 0.02:
                    pixels[i] = rng.randint(1, 30)

            f.write(bytes(pixels))


def generate_labels(path, seed=123):
    """Generate IDX1-UBYTE label file with 100 labels (digits 0-9)."""
    rng = random.Random(seed)

    with open(path, "wb") as f:
        # Header: magic=0x00000801, count (both big-endian)
        write_big_endian_uint32(f, 0x00000801)
        write_big_endian_uint32(f, NUM_IMAGES)

        # Generate labels: 10 of each digit for uniform distribution
        labels = list(range(10)) * 10
        rng.shuffle(labels)
        f.write(bytes(labels))


if __name__ == "__main__":
    generate_images("data/mini-images-idx3-ubyte")
    generate_labels("data/mini-labels-idx1-ubyte")
    print(f"Generated {NUM_IMAGES} images and labels in data/")

    # Verify the files
    with open("data/mini-images-idx3-ubyte", "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        count = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        print(f"Images: magic=0x{magic:08x}, count={count}, rows={rows}, cols={cols}")
        data = f.read()
        print(f"  Pixel data: {len(data)} bytes (expected {count * rows * cols})")

    with open("data/mini-labels-idx1-ubyte", "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        count = struct.unpack(">I", f.read(4))[0]
        print(f"Labels: magic=0x{magic:08x}, count={count}")
        labels = list(f.read())
        print(f"  Label distribution: {sorted(set(labels))} (unique values)")
        print(f"  First 20 labels: {labels[:20]}")
