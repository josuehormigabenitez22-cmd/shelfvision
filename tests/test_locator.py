"""Tests of the barcode locator on labels drawn on the fly, with no model."""

import numpy as np
import pytest

from shelfvision.code128 import CodeSet, encode, symbols_to_widths
from shelfvision.locator import Candidate, find_candidates, local_angle
from shelfvision.synthetic import LIGHT, render_label

MODULE_PX = 3.0
HEIGHT_PX = 60
WIDTHS = symbols_to_widths(encode([(CodeSet.B, "SHELF1")]))
CODE_WIDTH = sum(WIDTHS) * MODULE_PX
SIDE = 600
MAX_ANGLE_ERROR = 3.0
MAX_CENTER_ERROR = 10.0


def angle_error(measured: float, true: float) -> float:
    """Difference between two directions in degrees; a code read backwards is the same code."""
    return abs((measured - true + 90) % 180 - 90)


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0, 170.0])
def test_finds_one_label_with_its_angle_and_extent(angle):
    candidates = find_candidates(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, angle, SIDE))
    assert len(candidates) == 1
    (candidate,) = candidates
    assert np.hypot(candidate.center[0] - SIDE / 2, candidate.center[1] - SIDE / 2) < MAX_CENTER_ERROR
    assert angle_error(candidate.angle, angle) < MAX_ANGLE_ERROR
    assert 0.9 < candidate.width / CODE_WIDTH < 1.1


def test_two_close_labels_stay_separate():
    upper = render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE)
    image = np.minimum(np.roll(upper, -50, axis=0), np.roll(upper, 50, axis=0))
    assert len(find_candidates(image)) == 2


def test_blank_image_has_no_candidates():
    assert find_candidates(np.full((SIDE, SIDE), LIGHT)) == []


def test_center_outside_the_image_gives_no_angle():
    gray = np.zeros((1000, 1000), np.float32)
    assert local_angle(gray, (1195.0, 169.0), 162) is None
    assert local_angle(gray, (-1.0, 500.0), 162) is None


def test_label_cut_by_the_border_does_not_raise():
    image = np.roll(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=SIDE), SIDE // 2 - 20, axis=1)
    candidates = find_candidates(image)
    assert all(isinstance(candidate, Candidate) for candidate in candidates)


def test_rejects_a_color_image():
    with pytest.raises(ValueError, match="gray"):
        find_candidates(np.zeros((10, 10, 3), np.uint8))


def test_label_that_fills_most_of_the_image_is_found_whole():
    candidates = find_candidates(render_label(WIDTHS, MODULE_PX, HEIGHT_PX, side=round(1.2 * CODE_WIDTH)))
    assert len(candidates) == 1
    assert 0.9 < candidates[0].width / CODE_WIDTH < 1.1
