"""Tests of the votes between row blocks and of reading whole images, with labels drawn on the fly."""

from collections import Counter

import cv2
import numpy as np
import pytest

from shelfvision.barcodes import (
    MIN_BLOCK_ROWS,
    MIN_INTENSITY_SPAN,
    Stage,
    distinct,
    read_barcodes,
    read_candidate,
    row_blocks,
    winner,
    with_enough_contrast,
)
from shelfvision.code128 import CodeSet, encode, symbols_to_widths
from shelfvision.datatypes import Barcode
from shelfvision.locator import Candidate, find_candidates
from shelfvision.synthetic import DARK, LIGHT, ean13_widths, render_label

MODULE_PX = 3.0
HEIGHT_PX = 60
SIDE = 600
SHIFT = 90
# The axis is measured on the striped area, which the bars widen a little at its ends
MAX_AXIS_ERROR = 5.0


def label(text: str, angle: float = 0.0) -> np.ndarray:
    return render_label(symbols_to_widths(encode([(CodeSet.B, text)])), MODULE_PX, HEIGHT_PX, angle, SIDE)


@pytest.mark.parametrize(
    ("votes", "expected"),
    [
        ({}, None),
        ({"Box": 2}, None),
        ({"Box": 3}, "Box"),
        ({"Box": 3, "Bax": 1}, "Box"),
        ({"Box": 3, "Bax": 2}, None),
    ],
)
def test_winner_needs_enough_votes_and_a_clear_margin(votes, expected):
    assert winner(Counter(votes)) == expected


def test_row_blocks_cover_the_strip_at_each_height():
    strip = np.zeros((60, 10))
    sizes = [block.shape[0] for block in row_blocks(strip, (1, 1 / 3))]
    assert sizes == [60, 20, 20, 20]


def test_row_blocks_never_get_thinner_than_the_minimum():
    sizes = {block.shape[0] for block in row_blocks(np.zeros((20, 10)), (1 / 100,))}
    assert sizes == {MIN_BLOCK_ROWS}


@pytest.mark.parametrize("angle", [0.0, 60.0, 115.0])
def test_reads_a_located_label(angle):
    gray = label("Wrench", angle)
    (candidate,) = find_candidates(gray)
    assert read_candidate(gray, candidate) == "Wrench"


def test_costly_stage_is_skipped_without_earlier_votes():
    gray = label("Wrench")
    (candidate,) = find_candidates(gray)
    costly_only = (Stage(shears=8, heights=(1 / 8,), max_error=1.05, needs_votes=True),)
    assert read_candidate(gray, candidate, costly_only) is None


def test_noise_does_not_read():
    gray = np.random.default_rng(0).uniform(0, 255, (SIDE, SIDE)).astype(np.float32)
    assert read_candidate(gray, Candidate((SIDE / 2, SIDE / 2), 0.0, 300.0, 60.0)) is None


def test_reads_every_label_in_an_image_with_its_corners():
    gray = np.minimum(np.roll(label("Shoe"), -SHIFT, axis=0), np.roll(label("Mug"), SHIFT, axis=0))
    barcodes = sorted(read_barcodes(gray), key=lambda barcode: barcode.center[1])
    assert [barcode.value for barcode in barcodes] == ["Shoe", "Mug"]
    for barcode, row in zip(barcodes, (SIDE / 2 - SHIFT, SIDE / 2 + SHIFT), strict=True):
        assert barcode.center == pytest.approx((SIDE / 2, row), abs=10)


def test_blank_image_reads_nothing():
    assert read_barcodes(np.full((SIDE, SIDE), LIGHT)) == []


def leaning(image: np.ndarray, slope: float) -> np.ndarray:
    """Tilt the code's axis by slope while its bars stay vertical, as on a curved or oblique label."""
    matrix = np.float32([[1, 0, 0], [slope, 1, -slope * image.shape[1] / 2]])
    return cv2.warpAffine(image, matrix, image.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=LIGHT)


@pytest.mark.parametrize("slope", [-0.45, 0.3, 0.45])
def test_bars_leaning_against_the_axis_are_scanned_along_the_axis(slope):
    gray = leaning(label("Wrench"), slope)
    (candidate,) = find_candidates(gray)
    assert abs(candidate.angle - np.degrees(np.arctan(slope))) < MAX_AXIS_ERROR
    assert read_candidate(gray, candidate) == "Wrench"


def qr_code(text: str, module_px: int = 6, side: int = SIDE) -> np.ndarray:
    """A QR code with OpenCV's encoder, centered on a light square."""
    modules = cv2.QRCodeEncoder.create().encode(text)
    code = cv2.resize(modules, None, fx=module_px, fy=module_px, interpolation=cv2.INTER_NEAREST)
    image = np.full((side, side), LIGHT, dtype=np.float32)
    top, left = (side - code.shape[0]) // 2, (side - code.shape[1]) // 2
    image[top : top + code.shape[0], left : left + code.shape[1]] = np.where(code > 0, LIGHT, DARK)
    return image


@pytest.mark.parametrize("angle", [0.0, 35.0, 90.0])
@pytest.mark.parametrize("module_px", [2.0, 3.0])
def test_an_ean13_code_is_not_read_as_code128(angle, module_px):
    gray = render_label(ean13_widths("400638133393"), module_px, HEIGHT_PX, angle, SIDE)
    assert read_barcodes(gray) == []


@pytest.mark.parametrize("module_px", [4, 8])
def test_a_qr_code_is_not_read_as_code128(module_px):
    assert read_barcodes(qr_code("SHELF-2026", module_px)) == []


def test_next_to_other_codes_only_the_code128_is_read():
    others = np.minimum(np.roll(qr_code("SHELF-2026", 4), -SHIFT - 40, axis=0), label("Bottle"))
    ean = render_label(ean13_widths("871234567890"), 3.0, HEIGHT_PX, 0.0, SIDE)
    gray = np.minimum(others, np.roll(ean, SHIFT + 60, axis=0))
    assert [barcode.value for barcode in read_barcodes(gray)] == ["Bottle"]


def test_an_image_with_enough_contrast_is_left_as_it_is():
    gray = label("Box")
    assert with_enough_contrast(gray) is gray


def test_a_dark_image_is_stretched_to_the_minimum_span():
    stretched = with_enough_contrast(label("Box") * 0.3)
    low, high = np.percentile(stretched, (1, 99))
    assert high - low == pytest.approx(MIN_INTENSITY_SPAN)


def test_a_flat_image_is_left_as_it_is():
    flat = np.full((SIDE, SIDE), LIGHT, dtype=np.float32)
    assert with_enough_contrast(flat) is flat


def test_a_dark_photo_of_a_label_still_reads():
    assert [barcode.value for barcode in read_barcodes(label("Wrench") * 0.3)] == ["Wrench"]


def square(value: str, x: float) -> Barcode:
    return Barcode(value, ((x, 0.0), (x + 10, 0.0), (x + 10, 10.0), (x, 10.0)))


def test_a_label_read_by_two_candidates_counts_once_but_two_labels_with_one_value_stay():
    first, overlapping, elsewhere, other = square("A", 0), square("A", 5), square("A", 50), square("B", 2)
    assert distinct([first, overlapping, elsewhere, other]) == [first, elsewhere, other]


def test_of_two_overlapping_reads_of_one_label_the_tighter_outline_stays():
    wide = Barcode("A", ((0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)))
    tight = square("A", 100)
    assert distinct([wide, tight]) == [tight]
