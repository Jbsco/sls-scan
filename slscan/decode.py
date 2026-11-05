"""Decode: a captured Gray code stack and phase-shift stack to a float projector column
per camera pixel, with a mask of the pixels that decoded cleanly."""

import numpy as np

from . import patterns

SATURATED = 255


def decode_columns(gray: np.ndarray, phase: np.ndarray, width: int, pitch: int,
                   min_contrast: float = 10.0, min_modulation: float = 10.0,
                   max_disagreement: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the projector column per camera pixel and its mask.

    gray is the captured column stack in the order of patterns.gray_code_images, phase the
    captured stack of patterns.phase_shift_images. A pixel is kept when every Gray code
    pair has the contrast, no phase sample reaches SATURATED, the sinusoid has the
    modulation, and the phase column lies within max_disagreement of the Gray code column.
    """
    bits, mask = patterns.binarize_pairs(gray, min_contrast)
    columns = patterns.decode_gray_bits(bits)
    mask &= columns < width
    mask &= np.asarray(phase).max(axis=0) < SATURATED
    wrapped = patterns.wrapped_phase(phase)
    mask &= patterns.phase_modulation(phase) >= min_modulation
    result = patterns.unwrap_with_columns(wrapped, columns, pitch)
    mask &= np.abs(result - columns) <= max_disagreement
    mask &= (result >= -0.5) & (result <= width - 0.5)
    return result, mask

