"""Tests for the detector: the geometry runs without weights, the rest is marked slow."""

from pathlib import Path

import pytest
import torch

from shelfvision.detector import ObjectDetector, to_pixel_boxes
from shelfvision.images import load_image

IMAGE_PATH = Path(__file__).parents[1] / "examples" / "input" / "input_1.jpg"
TEMPLATES = ("a photo of a {}",)


def test_pixel_boxes_use_the_long_side_as_scale():
    boxes = to_pixel_boxes(torch.tensor([[0.5, 0.25, 0.5, 0.5]]), (200, 100))
    assert boxes.tolist() == [[50, 0, 150, 100]]


def test_pixel_boxes_are_clipped_to_the_image():
    boxes = to_pixel_boxes(torch.tensor([[0.0, 0.5, 0.5, 0.5]]), (200, 100))
    assert boxes.tolist() == [[0, 50, 50, 100]]


def test_box_inside_the_padding_becomes_empty():
    boxes = to_pixel_boxes(torch.tensor([[0.5, 0.9, 0.2, 0.1]]), (200, 100))
    assert boxes[0, 1] == boxes[0, 3] == 100


def test_creating_a_detector_does_not_load_the_model():
    detector = ObjectDetector()
    assert "_model" not in vars(detector)
    assert "_processor" not in vars(detector)


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector()


@pytest.fixture(scope="module")
def features(detector):
    return detector.encode_image(load_image(IMAGE_PATH))


@pytest.mark.slow
def test_features_have_one_box_per_token_inside_the_image(features):
    width, height = features.image_size
    assert features.tokens.shape[1] == features.boxes.shape[0] == features.objectness.shape[0]
    assert features.boxes[:, [0, 2]].max() <= width
    assert features.boxes[:, [1, 3]].max() <= height
    assert features.boxes.min() >= 0


@pytest.mark.slow
def test_name_embeddings_are_unit_vectors_averaged_per_name(detector):
    embeddings = detector.encode_names(("box", "mug"), ("a {}", "a photo of a {}"))
    assert embeddings.shape[0] == 2
    assert embeddings.norm(dim=-1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.slow
def test_repeated_names_come_from_the_cache(detector):
    first = detector.encode_names(("box",), TEMPLATES)
    assert detector.encode_names(("box",), TEMPLATES) is first


@pytest.mark.slow
def test_score_has_one_probability_per_box_and_name(detector, features):
    scores = detector.score(features, detector.encode_names(("box", "mug"), TEMPLATES))
    assert scores.shape == (features.boxes.shape[0], 2)
    assert 0 <= scores.min() <= scores.max() <= 1
