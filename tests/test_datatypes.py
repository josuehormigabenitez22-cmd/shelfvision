"""Tests for the value types and the normalization of user queries."""

import dataclasses

import pytest

from shelfvision.datatypes import ALL_OBJECTS, Barcode, Box, DetectedObject, normalize_query


def test_box_area():
    assert Box(1, 2, 4, 6).area == 12


def test_box_rejects_swapped_corners():
    with pytest.raises(ValueError, match="out of order"):
        Box(5, 0, 1, 1)


@pytest.mark.parametrize(
    ("point", "expected"),
    [((5, 5), True), ((0, 0), True), ((10, 10), True), ((10.1, 5), False), ((5, -1), False)],
)
def test_box_contains_includes_borders(point, expected):
    assert Box(0, 0, 10, 10).contains(point) is expected


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (Box(0, 0, 10, 10), 1.0),
        (Box(5, 0, 15, 10), 50 / 150),
        (Box(20, 20, 30, 30), 0.0),
        (Box(10, 0, 20, 10), 0.0),
    ],
)
def test_box_iou(other, expected):
    box = Box(0, 0, 10, 10)
    assert box.iou(other) == pytest.approx(expected)
    assert other.iou(box) == pytest.approx(expected)


def test_iou_of_two_empty_boxes_is_zero():
    assert Box(1, 1, 1, 1).iou(Box(1, 1, 1, 1)) == 0.0


def test_barcode_box_and_center_come_from_its_corners():
    barcode = Barcode("Shoe", ((2, 1), (6, 3), (4, 7), (0, 5)))
    assert barcode.box == Box(0, 1, 6, 7)
    assert barcode.center == (3, 4)
    assert barcode.normal is None


def test_values_are_immutable():
    detected = DetectedObject("box", Box(0, 0, 1, 1), 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        detected.score = 0.9


@pytest.mark.parametrize("query", ["all", "ALL", " All ", ["all"], ["box", "ALL"]])
def test_every_spelling_of_all_selects_all_objects(query):
    assert normalize_query(query) == ALL_OBJECTS


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("box", ("box",)),
        (["box", "mug"], ("box", "mug")),
        (("Box", " MUG "), ("box", "mug")),
        (["cell   phone"], ("cell phone",)),
        (["mug", "box", "MUG", " mug"], ("mug", "box")),
        ((name for name in ["box"]), ("box",)),
        (["allen key"], ("allen key",)),
    ],
)
def test_names_are_cleaned_and_keep_their_order(query, expected):
    assert normalize_query(query) == expected


@pytest.mark.parametrize("query", [[], (), "", "   ", ["", "  "]])
def test_empty_queries_select_nothing(query):
    assert normalize_query(query) == ()


@pytest.mark.parametrize("query", [None, 5, ["box", 5], [None]])
def test_non_string_queries_are_rejected(query):
    with pytest.raises(TypeError):
        normalize_query(query)
