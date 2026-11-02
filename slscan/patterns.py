"""Pattern generation: Gray code bit planes and phase-shifted sinusoids.

The Gray code gives every projector column an unambiguous index, one bit plane per
image, each shown with its inverse. The phase shift refines the column inside one
period to a fraction of a pixel. The decode tables here are pure arithmetic and are
tested in isolation of any camera or display.
"""

import math

import numpy as np


def gray_encode(n: int) -> int:
    """Return the Gray code of a non-negative integer."""
    if n < 0:
        raise ValueError("gray_encode takes a non-negative integer")
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    """Return the integer whose Gray code is g."""
    if g < 0:
        raise ValueError("gray_decode takes a non-negative integer")
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def bit_planes(width: int) -> int:
    """Return the number of bit planes that index every column of a projector."""
    if width < 1:
        raise ValueError("width must be at least 1")
    return max(1, math.ceil(math.log2(width)))


def gray_code_images(width: int, height: int) -> np.ndarray:
    """Return the Gray code column patterns as a uint8 stack.

    Shape is (2 * planes, height, width): plane k at index 2k and its inverse at
    2k + 1, most significant plane first. Values are 0 and 255.
    """
    planes = bit_planes(width)
    columns = np.arange(width, dtype=np.int64)
    codes = columns ^ (columns >> 1)
    stack = np.empty((2 * planes, height, width), dtype=np.uint8)
    for k in range(planes):
        shift = planes - 1 - k
        bits = ((codes >> shift) & 1).astype(np.uint8) * 255
        row = np.broadcast_to(bits, (height, width))
        stack[2 * k] = row
        stack[2 * k + 1] = 255 - row
    return stack


def decode_gray_bits(bits: np.ndarray) -> np.ndarray:
    """Decode a stack of binary planes, most significant first, to column indices.

    bits has shape (planes, ...) with values 0 or 1. The result has the trailing shape
    and holds the column index of each element.
    """
    bits = np.asarray(bits)
    if bits.ndim < 1 or bits.shape[0] < 1:
        raise ValueError("bits needs at least one plane")
    gray = np.zeros(bits.shape[1:], dtype=np.int64)
    for plane in bits:
        gray = (gray << 1) | plane.astype(np.int64)
    n = gray.copy()
    shift = gray >> 1
    while np.any(shift):
        n ^= shift
        shift >>= 1
    return n


def phase_shift_images(width: int, height: int, pitch: int, steps: int = 4) -> np.ndarray:
    """Return steps sinusoidal column patterns of the given pitch as a uint8 stack.

    Pattern i has phase 2 * pi * i / steps. Shape is (steps, height, width).
    """
    if pitch < 2:
        raise ValueError("pitch must be at least 2 projector pixels")
    if steps < 3:
        raise ValueError("phase shift needs at least 3 steps")
    columns = np.arange(width, dtype=np.float64)
    stack = np.empty((steps, height, width), dtype=np.uint8)
    for i in range(steps):
        phase = 2.0 * math.pi * (columns / pitch + i / steps)
        row = np.round(127.5 + 127.5 * np.cos(phase)).astype(np.uint8)
        stack[i] = np.broadcast_to(row, (height, width))
    return stack


def wrapped_phase(samples: np.ndarray) -> np.ndarray:
    """Return the wrapped phase in [0, 2 pi) from N phase-shifted intensity samples.

    samples has shape (steps, ...), sample i taken with pattern phase 2 * pi * i / steps.
    """
    samples = np.asarray(samples, dtype=np.float64)
    steps = samples.shape[0]
    if steps < 3:
        raise ValueError("phase shift needs at least 3 steps")
    i = np.arange(steps, dtype=np.float64).reshape((steps,) + (1,) * (samples.ndim - 1))
    angles = 2.0 * math.pi * i / steps
    numerator = np.sum(samples * np.sin(angles), axis=0)
    denominator = np.sum(samples * np.cos(angles), axis=0)
    return np.mod(-np.arctan2(numerator, denominator), 2.0 * math.pi)


def unwrap_columns(phase: np.ndarray, period_index: np.ndarray, pitch: int) -> np.ndarray:
    """Combine a wrapped phase and a Gray code period index into a float column."""
    return period_index.astype(np.float64) * pitch + phase * pitch / (2.0 * math.pi)
