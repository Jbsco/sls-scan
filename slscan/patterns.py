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


def gray_code_images(width: int, height: int, axis: str = "columns") -> np.ndarray:
    """Return the Gray code patterns of one axis as a uint8 stack.

    axis "columns" codes each projector column, "rows" each projector row. Shape is
    (2 * planes, height, width): plane k at index 2k and its inverse at 2k + 1, most
    significant plane first. Values are 0 and 255.
    """
    if axis not in ("columns", "rows"):
        raise ValueError('axis is "columns" or "rows"')
    length = width if axis == "columns" else height
    planes = bit_planes(length)
    index = np.arange(length, dtype=np.int64)
    codes = index ^ (index >> 1)
    stack = np.empty((2 * planes, height, width), dtype=np.uint8)
    for k in range(planes):
        shift = planes - 1 - k
        bits = ((codes >> shift) & 1).astype(np.uint8) * 255
        line = bits if axis == "columns" else bits[:, np.newaxis]
        image = np.broadcast_to(line, (height, width))
        stack[2 * k] = image
        stack[2 * k + 1] = 255 - image
    return stack


def binarize_pairs(captured: np.ndarray, min_contrast: float) -> tuple[np.ndarray, np.ndarray]:
    """Binarize captured pattern and inverse pairs.

    captured has shape (2 * planes, ...) in the order of gray_code_images. Returns the
    bit planes (planes, ...) as uint8 and a mask of the elements whose every pair
    differs by at least min_contrast.
    """
    captured = np.asarray(captured)
    if captured.ndim < 1 or captured.shape[0] < 2 or captured.shape[0] % 2:
        raise ValueError("captured needs pattern and inverse pairs")
    planes = captured.shape[0] // 2
    bits = np.empty((planes,) + captured.shape[1:], dtype=np.uint8)
    mask = np.ones(captured.shape[1:], dtype=bool)
    for k in range(planes):
        diff = captured[2 * k].astype(np.float32) - captured[2 * k + 1].astype(np.float32)
        bits[k] = diff > 0
        mask &= np.abs(diff) >= min_contrast
    return bits, mask


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


def phase_modulation(samples: np.ndarray) -> np.ndarray:
    """Return the amplitude of the sinusoid in N phase-shifted samples, in sample units."""
    samples = np.asarray(samples, dtype=np.float64)
    steps = samples.shape[0]
    if steps < 3:
        raise ValueError("phase shift needs at least 3 steps")
    i = np.arange(steps, dtype=np.float64).reshape((steps,) + (1,) * (samples.ndim - 1))
    angles = 2.0 * math.pi * i / steps
    numerator = np.sum(samples * np.sin(angles), axis=0)
    denominator = np.sum(samples * np.cos(angles), axis=0)
    return 2.0 / steps * np.hypot(numerator, denominator)


def unwrap_with_columns(phase: np.ndarray, columns: np.ndarray, pitch: int) -> np.ndarray:
    """Return the float column nearest the Gray code column with the given wrapped phase.

    The phase fixes the position inside a period and the Gray code column picks the
    period, so a Gray code error under half a period does not move the result.
    """
    fraction = phase * pitch / (2.0 * math.pi)
    return fraction + pitch * np.round((columns - fraction) / pitch)


def unwrap_columns(phase: np.ndarray, period_index: np.ndarray, pitch: int) -> np.ndarray:
    """Combine a wrapped phase and a Gray code period index into a float column."""
    return period_index.astype(np.float64) * pitch + phase * pitch / (2.0 * math.pi)
