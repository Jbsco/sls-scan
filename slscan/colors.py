"""Point colors from a mono camera under a projector's black field and its full red,
green and blue fields. Each field less the black one removes room light, and a white
card's response per channel, its white balance, turns the rest into reflectance."""

import numpy as np

CHANNELS = ("red", "green", "blue")


def responses(view: dict) -> np.ndarray:
    """Return each pixel's red, green and blue field less the black one, (H, W, 3)."""
    black = view["black"].astype(np.float64)
    return np.stack([view[c].astype(np.float64) - black for c in CHANNELS], axis=-1)


def white_balance(view: dict, floor: float = 20.0) -> np.ndarray:
    """Return each channel's median response over the pixels of a white card's view whose
    every channel responds above floor grey levels."""
    r = responses(view)
    lit = r.min(axis=-1) > floor
    if not lit.any():
        raise ValueError("no pixel of the white card responds in every channel")
    return np.median(r[lit], axis=0)


def point_colors(view: dict, pixels: np.ndarray, white=None) -> np.ndarray:
    """Return red, green and blue bytes (N, 3) for camera pixels (N, 2) as x, y: each
    channel's response over the white balance, 255 for white, or the raw response in grey
    levels without one."""
    at = np.rint(pixels).astype(np.int64)
    r = responses(view)[at[:, 1], at[:, 0]]
    if white is not None:
        r = 255.0 * r / np.asarray(white, dtype=np.float64)
    return np.clip(np.rint(r), 0, 255).astype(np.uint8)
