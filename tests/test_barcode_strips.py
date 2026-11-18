"""Tests of the strips cut around candidates and of the shear search, on labels drawn on the fly."""

import cv2
import numpy as np
import pytest

from shelfvision.barcodes import (
    MAX_SHEAR,
    MIN_STRIP_ROWS,
    SHEAR_STEPS,
    aligned_strip,
    best_read,
    ranked_shears,
    read_profile,
    strip_size,
)
from shelfvision.code128 import CodeSet, encode, symbols_to_widths
from shelfvision.locator import Candidate, find_candidates
from shelfvision.synthetic import LIGHT, render_label

TEXT = "SHELF1"
MODULE_PX = 3.0
HEIGHT_PX = 60
SIDE = 600
WIDTHS = symbols_to_widths(encode([(CodeSet.B, TEXT)]))
CODE_WIDTH = sum(WIDTHS) * MODULE_PX
SHEAR_STEP = 2 * MAX_SHEAR / (SHEAR_STEPS - 1)


def slanted(image: np.ndarray, slope: float) -> np.ndarray:
    """Slant vertical bars so that x moves by slope * y around the middle row."""
    matrix = np.float32([[1, slope, -slope * image.shape[0] / 2], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, image.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=LIGHT)


def centered_candidate() -> Candidate:
    return Candidate((SIDE / 2, SIDE / 2), 0.0, CODE_WIDTH, HEIGHT_PX)


def read_best_strip(gray: np.ndarray, candidate: Candidate) -> str | None:
    strip = aligned_strip(gray, candidate, ranked_shears(gray, candidate)[0])
    reading = best_read(read_profile(strip.mean(axis=0)))
    return reading.text if reading else None


def test_strip_holds_the_quiet_zones_and_enough_rows():
    columns, rows = strip_size(Candidate((0.0, 0.0), 0.0, 100.0, 2.0))
    assert columns > 100
    assert rows == MIN_STRIP_ROWS


@pytest.mark.parametrize("slope", [-0.3, -0.1, 0.0, 0.2, 0.35])
def test_shear_search_undoes_slanted_bars(slope):
    gray = slanted(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE), slope)
    best = ranked_shears(gray, centered_candidate())[0]
    assert best == pytest.approx(-slope, abs=SHEAR_STEP)


def test_slanted_bars_read_only_after_the_shear_search():
    gray = slanted(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE), 0.3)
    straight = aligned_strip(gray, centered_candidate(), 0.0)
    assert best_read(read_profile(straight.mean(axis=0))) is None
    assert read_best_strip(gray, centered_candidate()) == TEXT


@pytest.mark.parametrize("angle", [0.0, 45.0, 100.0, 160.0])
def test_located_label_reads_from_its_strip(angle):
    gray = render_label(WIDTHS, MODULE_PX, HEIGHT_PX, angle, SIDE)
    (candidate,) = find_candidates(gray)
    assert read_best_strip(gray, candidate) == TEXT


def test_candidate_at_the_image_border_does_not_raise():
    gray = render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE)
    strip = aligned_strip(gray, Candidate((5.0, 5.0), 30.0, CODE_WIDTH, HEIGHT_PX), MAX_SHEAR)
    assert strip.shape == strip_size(centered_candidate())[::-1]


def test_shear_search_is_centered_on_the_expected_slant():
    gray = slanted(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE), -0.6)
    candidate = Candidate((SIDE / 2, SIDE / 2), 0.0, CODE_WIDTH, HEIGHT_PX, slant=0.5)
    assert ranked_shears(gray, candidate)[0] == pytest.approx(0.6, abs=SHEAR_STEP)
