"""Outlines of objects: geometry, masks to polygons, the links, the JSON report and the drawing."""

import numpy as np
import pytest
from PIL import Image

from shelfvision.datatypes import Barcode, Box, DetectedObject, Outline
from shelfvision.draw import OUTLINE_ALPHA, PALETTE, draw_objects
from shelfvision.export import object_record
from shelfvision.relations import owner
from shelfvision.segmenter import Segmenter, mask_outline

SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
TRIANGLE = ((20.0, 0.0), (30.0, 0.0), (20.0, 10.0))
BIG_SQUARE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))


def code_at(x: float, y: float) -> Barcode:
    return Barcode("A", ((x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1), (x - 1, y + 1)))


def test_an_outline_adds_the_areas_of_its_polygons() -> None:
    assert Outline((SQUARE,)).area == pytest.approx(100)
    assert Outline((SQUARE, TRIANGLE)).area == pytest.approx(150)


def test_an_outline_contains_the_points_inside_any_of_its_polygons() -> None:
    outline = Outline((SQUARE, TRIANGLE))
    assert outline.contains((5, 5))
    assert outline.contains((22, 2))
    assert not outline.contains((28, 8))
    assert not outline.contains((15, 5))


def test_a_mask_becomes_its_outer_polygon_without_specks() -> None:
    mask = np.zeros((100, 200), bool)
    mask[20:80, 30:170] = True
    mask[5, 5] = True
    outline = mask_outline(mask, tolerance=1.0)
    assert outline is not None
    (polygon,) = outline.polygons
    assert Box.around(polygon) == Box(30, 20, 169, 79)
    assert outline.contains((100, 50))


def test_a_mask_with_two_large_parts_keeps_both_and_an_empty_mask_gives_none() -> None:
    mask = np.zeros((100, 200), bool)
    mask[10:40, 10:40] = mask[60:90, 150:190] = True
    outline = mask_outline(mask, tolerance=1.0)
    assert outline is not None and len(outline.polygons) == 2
    assert mask_outline(np.zeros((10, 10), bool), tolerance=1.0) is None


def test_task_3_prefers_the_outline_that_holds_the_code_to_a_smaller_box() -> None:
    # The code lies in both boxes; the small box's object does not cover it, the large one does
    small = DetectedObject("cup", Box(0, 0, 50, 50), 0.5, Outline((SQUARE,)))
    large = DetectedObject("box", Box(0, 0, 100, 100), 0.5, Outline((BIG_SQUARE,)))
    assert owner(code_at(40, 40), [small, large]) is large
    assert owner(code_at(5, 5), [small, large]) is small


def test_task_3_falls_back_to_the_boxes_when_no_outline_holds_the_code() -> None:
    outlined = DetectedObject("cup", Box(0, 0, 50, 50), 0.5, Outline((SQUARE,)))
    boxed = DetectedObject("box", Box(0, 0, 100, 100), 0.5)
    assert owner(code_at(40, 40), [outlined, boxed]) is outlined
    assert owner(code_at(80, 80), [outlined, boxed]) is boxed


def test_the_report_has_an_outline_only_with_segmentation() -> None:
    plain = DetectedObject("cup", Box(0, 0, 10, 10), 0.5)
    assert "outline" not in object_record(plain)
    outlined = DetectedObject("cup", Box(0, 0, 10, 10), 0.5, Outline((((0.123, 0), (10, 0), (10, 10)),)))
    assert object_record(outlined)["outline"] == [[[0.12, 0.0], [10.0, 0.0], [10.0, 10.0]]]


def test_the_outline_is_filled_in_the_object_color_and_nothing_else_changes() -> None:
    image = Image.new("RGB", (200, 200), (255, 255, 255))
    corner = ((120.0, 120.0), (190.0, 120.0), (190.0, 190.0), (120.0, 190.0))
    item = DetectedObject("cup", Box(110, 110, 195, 195), 0.5, Outline((corner,)))
    pixels = np.asarray(draw_objects(image, [item])).astype(int)
    plain = np.asarray(draw_objects(image, [DetectedObject("cup", item.box, 0.5)])).astype(int)
    alpha = OUTLINE_ALPHA / 255
    assert pixels[150, 150] == pytest.approx([255 * (1 - alpha) + c * alpha for c in PALETTE[0]], abs=1)
    assert (plain[150, 150] == 255).all()
    assert (pixels[5:100, 150] == plain[5:100, 150]).all()


@pytest.mark.slow
def test_sam_outlines_a_dark_rectangle_from_its_box() -> None:
    image = Image.new("RGB", (640, 480), (235, 235, 235))
    image.paste((40, 60, 160), (200, 150, 440, 330))
    segmenter = Segmenter("cpu")
    (outline,) = segmenter.outlines(segmenter.encode_image(image), [Box(180, 130, 460, 350)])
    assert outline is not None
    assert outline.area == pytest.approx(240 * 180, rel=0.1)
    assert outline.contains((320, 240)) and not outline.contains((190, 140))
