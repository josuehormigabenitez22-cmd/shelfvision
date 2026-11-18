"""Normals and 3D arrows on hand-made normal maps; only the tests marked slow load MoGe."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shelfvision.datatypes import Barcode
from shelfvision.images import load_image
from shelfvision.normals import (
    ARROW_LABEL_WIDTHS,
    Intrinsics,
    NormalEstimator,
    SurfaceMap,
    arrow_tip,
    facing_camera,
    label_width,
    region_normal,
    with_normals,
)

CAMERA = Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=400.0)
TOWARDS_CAMERA = (0.0, 0.0, -1.0)
EXAMPLE = Path(__file__).parents[1] / "examples" / "input" / "input_1.jpg"


def label(center: tuple[float, float], width: float, normal=None) -> Barcode:
    x, y = center
    half = width / 2
    corners = (x - half, y - 10), (x + half, y - 10), (x + half, y + 10), (x - half, y + 10)
    return Barcode("A1", corners, normal)


def flat_map(normal, rows: int = 400, cols: int = 500) -> np.ndarray:
    return np.tile(np.asarray(normal, dtype=np.float32), (rows, cols, 1))


def test_ray_and_project_are_inverse():
    point = (730.0, 215.0)
    assert np.allclose(CAMERA.project(3.0 * CAMERA.ray(point)), point)


def test_region_normal_is_the_unit_median_inside_the_polygon():
    normals = flat_map((1.0, 0.0, 0.0))
    normals[100:200, 100:200] = (0.0, 0.6, -0.8)
    # The polygon is in original pixels, twice the size of the map
    polygon = ((220, 220), (380, 220), (380, 380), (220, 380))
    assert np.allclose(region_normal(normals, polygon, 0.5), (0.0, 0.6, -0.8))


def test_region_normal_outside_the_map_is_none():
    assert region_normal(flat_map(TOWARDS_CAMERA), ((900, 900), (950, 900), (950, 950)), 1.0) is None


def test_a_normal_pointing_away_is_flipped_towards_the_camera():
    ray = CAMERA.ray((500.0, 400.0))
    assert np.allclose(facing_camera(np.array([0.0, 0.0, 1.0]), ray), TOWARDS_CAMERA)
    assert np.allclose(facing_camera(np.array(TOWARDS_CAMERA), ray), TOWARDS_CAMERA)


def test_with_normals_sets_unit_normals_facing_the_camera_and_none_off_the_map():
    surface = SurfaceMap(flat_map((0.0, 0.0, 1.0)), 1.0, CAMERA)
    inside, outside = with_normals([label((250, 200), 80), label((2000, 2000), 80)], surface)
    assert np.allclose(inside.normal, TOWARDS_CAMERA)
    assert outside.normal is None
    assert inside.value == "A1" and inside.corners == label((250, 200), 80).corners


def test_label_width_is_the_longest_side():
    assert label_width(label((300, 300), 120)) == pytest.approx(120)


def test_an_arrow_facing_the_camera_ends_on_the_label_center():
    assert np.allclose(arrow_tip(label((500, 400), 100, TOWARDS_CAMERA), CAMERA), (500, 400))


def test_a_sideways_arrow_at_the_center_is_a_fixed_number_of_label_widths_long():
    tip = arrow_tip(label((500, 400), 100, (1.0, 0.0, 0.0)), CAMERA)
    assert np.allclose(tip, (500 + ARROW_LABEL_WIDTHS * 100, 400))


def test_no_normal_gives_no_arrow():
    assert arrow_tip(label((500, 400), 100), CAMERA) is None


def test_without_moge_only_the_normals_fail_with_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "moge.model.v2", None)
    with pytest.raises(ImportError, match="Windows and Linux"):
        NormalEstimator().estimate(Image.new("RGB", (64, 36)))


@pytest.mark.slow
def test_moge_gives_unit_normals_facing_the_camera_and_a_plausible_focal():
    image = load_image(EXAMPLE)
    surface = NormalEstimator().estimate(image)
    width, height = image.size
    assert surface.normals.shape[1] == round(width * surface.scale)
    assert np.allclose(np.linalg.norm(surface.normals, axis=2), 1.0, atol=1e-3)
    assert 0.5 * width < surface.intrinsics.fx < 2.0 * width
    (box_label,) = with_normals([label((width / 2, height * 0.6), 200)], surface)
    assert np.linalg.norm(box_label.normal) == pytest.approx(1.0)
    assert box_label.normal[2] < 0
