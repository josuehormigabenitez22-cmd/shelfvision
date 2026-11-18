"""Synthetic gray lines of barcodes, only for tests and benchmarks"""

from collections.abc import Sequence
from itertools import groupby
from typing import Final

import cv2
import numpy as np

# Samples per pixel when drawing a line, so that an edge can fall anywhere inside a pixel
SUPERSAMPLE: Final = 16
QUIET_MODULES: Final = 10
DARK: Final = 30.0
LIGHT: Final = 220.0
# EAN-13, another common 1D code: a reader of Code128 must not take it for one of its own
EAN_LEFT_ODD: Final = ("0001101", "0011001", "0010011", "0111101", "0100011",
                       "0110001", "0101111", "0111011", "0110111", "0001011")
EAN_PARITIES: Final = ("LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
                       "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL")


def render_profile(
    widths: Sequence[int],
    module_px: float,
    blur_px: float = 0.0,
    ink_px: float = 0.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the pixels of a line that crosses a code: ink_px widens each bar, as printing does."""
    edges = QUIET_MODULES * module_px + np.cumsum([0, *widths]) * module_px
    length = round(edges[-1] + QUIET_MODULES * module_px)
    samples = (np.arange(length * SUPERSAMPLE) + 0.5) / SUPERSAMPLE
    line = np.full(samples.size, LIGHT, dtype=np.float32)
    for bar_start, bar_end in zip(edges[:-1:2], edges[1::2], strict=True):
        line[(samples >= bar_start - ink_px / 2) & (samples < bar_end + ink_px / 2)] = DARK
    if blur_px > 0:
        line = cv2.GaussianBlur(line.reshape(-1, 1), (1, 0), blur_px * SUPERSAMPLE).ravel()
    pixels = line.reshape(-1, SUPERSAMPLE).mean(axis=1)
    return pixels if rng is None else pixels + rng.normal(0.0, noise, pixels.size)


def render_label(
    widths: Sequence[int], module_px: float, height_px: int, angle: float = 0.0, side: int = 600
) -> np.ndarray:
    """Return a square gray image with a code at its center; its scan line points at angle degrees."""
    label = np.tile(render_profile(widths, module_px), (height_px, 1))
    image = np.full((side, side), LIGHT, dtype=np.float32)
    top, left = (side - label.shape[0]) // 2, (side - label.shape[1]) // 2
    image[top : top + label.shape[0], left : left + label.shape[1]] = label
    # OpenCV turns counterclockwise on screen, and image rows grow downwards, hence the minus sign
    turn = cv2.getRotationMatrix2D((side / 2, side / 2), -angle, 1.0)
    return cv2.warpAffine(image, turn, (side, side), flags=cv2.INTER_LINEAR, borderValue=LIGHT)


def ean13_widths(twelve_digits: str) -> tuple[int, ...]:
    """Return the bar and space widths, in modules, of the EAN-13 code of twelve digits plus its check."""
    weights = (1, 3) * 6
    check = -sum(int(digit) * weight for digit, weight in zip(twelve_digits, weights, strict=True)) % 10
    first, left, right = twelve_digits[0], twelve_digits[1:7], twelve_digits[7:] + str(check)
    inverted = [pattern.translate(str.maketrans("01", "10")) for pattern in EAN_LEFT_ODD]
    halves = (
        EAN_LEFT_ODD[int(digit)] if parity == "L" else inverted[int(digit)][::-1]
        for digit, parity in zip(left, EAN_PARITIES[int(first)], strict=True)
    )
    bits = "101" + "".join(halves) + "01010" + "".join(inverted[int(digit)] for digit in right) + "101"
    return tuple(len(list(run)) for _, run in groupby(bits))
