"""Tests for drawing, using small images created on the fly and no model."""

from itertools import combinations

import numpy as np
import pytest
from PIL import Image

from shelfvision.datatypes import Barcode, Box, DetectedObject
from shelfvision.draw import (
    ARROW_COLOR,
    MIN_FONT_SIZE,
    MIN_LINE_WIDTH,
    TEXT_COLOR,
    add_banner,
    arrow_head,
    describe_query,
    draw_barcodes,
    draw_objects,
    place_label,
    render_barcodes,
    render_detections,
    style_for,
    wrap_text,
)

BACKGROUND = (200, 200, 200)
LABEL_SIZE = (40, 10)
IMAGE_SIZE = (200, 100)
BOX = Box(50, 50, 150, 90)


def blank(size=IMAGE_SIZE):
    return Image.new("RGB", size, BACKGROUND)


def test_sizes_grow_with_the_image():
    small, large = style_for((640, 360)), style_for((3840, 2160))
    assert large.font.size > small.font.size
    assert large.line_width > small.line_width


def test_tiny_image_keeps_a_readable_font():
    style = style_for((20, 10))
    assert style.font.size == MIN_FONT_SIZE
    assert style.line_width == MIN_LINE_WIDTH


def test_label_goes_above_the_box_when_there_is_room():
    assert place_label(BOX, LABEL_SIZE, IMAGE_SIZE, []) == Box(50, 40, 90, 50)


def test_label_of_a_box_at_the_top_goes_inside_it():
    assert place_label(Box(50, 0, 150, 90), LABEL_SIZE, IMAGE_SIZE, []) == Box(50, 0, 90, 10)


def test_label_of_a_box_at_the_right_stays_inside_the_image():
    label = place_label(Box(190, 50, 200, 90), LABEL_SIZE, IMAGE_SIZE, [])
    assert label.x1 == IMAGE_SIZE[0]


def test_label_avoids_the_labels_already_placed():
    taken = [Box(50, 40, 90, 50)]
    assert place_label(BOX, LABEL_SIZE, IMAGE_SIZE, taken) == Box(50, 50, 90, 60)


def test_many_boxes_in_the_same_place_get_separate_labels():
    taken = []
    for _ in range(4):
        taken.append(place_label(BOX, LABEL_SIZE, IMAGE_SIZE, taken))
    assert all(first.iou(second) == 0 for first, second in combinations(taken, 2))


def test_label_larger_than_the_image_gives_no_error():
    label = place_label(BOX, (500, 500), IMAGE_SIZE, [])
    assert (label.x0, label.y0) == (0, 0)


def test_drawing_does_not_change_the_original():
    image = blank()
    result = draw_objects(image, [DetectedObject("box", BOX, 0.5)])
    assert image.getpixel((50, 70)) == BACKGROUND
    assert result.getpixel((50, 70)) != BACKGROUND


def test_same_name_has_the_same_color_and_other_names_do_not():
    objects = [
        DetectedObject("box", Box(10, 30, 20, 90), 0.5),
        DetectedObject("box", Box(40, 30, 50, 90), 0.5),
        DetectedObject("mug", Box(70, 30, 80, 90), 0.5),
    ]
    result = draw_objects(blank(), objects)
    assert result.getpixel((10, 80)) == result.getpixel((40, 80)) != result.getpixel((70, 80))


def test_no_objects_leave_the_image_unchanged():
    assert draw_objects(blank(), []).tobytes() == blank().tobytes()


def test_banner_only_adds_height_and_hides_nothing():
    image = blank()
    result = add_banner(image, "Prompt: box")
    assert result.width == image.width
    banner_height = result.height - image.height
    assert banner_height > 0
    assert result.crop((0, banner_height, *result.size)).tobytes() == image.tobytes()


def test_banner_has_visible_text():
    result = add_banner(blank(), "Prompt")
    assert TEXT_COLOR in {color for _, color in result.getcolors(maxcolors=100000)}


def test_long_prompt_wraps_inside_the_image():
    font = style_for(IMAGE_SIZE).font
    lines = wrap_text("shoe bottle wrench screwdriver scissors mug box", font, 100)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 100 for line in lines)


def test_word_wider_than_the_image_gives_no_error():
    font = style_for(IMAGE_SIZE).font
    assert wrap_text("screwdriver", font, 1) == ["screwdriver"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [("ALL", "all"), (["Box", "mug"], "box, mug"), ([], "(empty)")],
)
def test_query_is_shown_as_the_user_should_read_it(query, expected):
    assert describe_query(query) == expected


def test_render_gives_equal_images_for_equal_inputs():
    objects = [DetectedObject("box", BOX, 0.5)]
    first = render_detections(blank(), objects, ["box"])
    second = render_detections(blank(), objects, ["box"])
    assert first.size[1] > IMAGE_SIZE[1]
    assert first.tobytes() == second.tobytes()


LABEL = Barcode("Kz9", ((60, 60), (100, 60), (100, 70), (60, 70)))


def arrow_pixels(image):
    return int((np.asarray(image) == ARROW_COLOR).all(axis=2).sum())


def test_arrow_head_points_at_the_tip_and_is_symmetric():
    tip, left, right = arrow_head((0, 0), (10, 0), 4)
    assert tip == (10, 0)
    assert left == pytest.approx((6, 2)) and right == pytest.approx((6, -2))


def test_a_zero_arrow_has_no_head():
    assert arrow_head((5, 5), (5, 5), 4) is None


def test_barcode_drawing_shows_the_arrow_along_its_way_and_keeps_the_original():
    image = blank()
    result = draw_barcodes(image, [LABEL], [(180, 20)])
    assert image.tobytes() == blank().tobytes()
    assert result.getpixel((140, 40)) == ARROW_COLOR


def test_a_barcode_without_normal_gets_only_the_start_dot():
    with_arrow = arrow_pixels(draw_barcodes(blank(), [LABEL], [(180, 20)]))
    without = arrow_pixels(draw_barcodes(blank(), [LABEL], [None]))
    assert 0 < without < with_arrow


def test_every_barcode_needs_its_tip():
    with pytest.raises(ValueError):
        draw_barcodes(blank(), [LABEL], [])


def test_barcode_render_gives_equal_images_for_equal_inputs():
    first = render_barcodes(blank(), [LABEL], [(180, 20)], "all")
    second = render_barcodes(blank(), [LABEL], [(180, 20)], "all")
    assert first.size[1] > IMAGE_SIZE[1]
    assert first.tobytes() == second.tobytes()
