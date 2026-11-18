"""Tests for the "all" and query modes: the selection rules run without weights, the rest is slow."""

from pathlib import Path

import pytest
import torch

from shelfvision.detector import (
    MAX_RIVAL_SIMILARITY,
    MIN_NAME_SCORE,
    UNKNOWN_NAME,
    VOCABULARY,
    ObjectDetector,
    display_name,
    find_rivals,
    select_matches,
    select_objects,
)
from shelfvision.images import load_image

IMAGE_PATH = Path(__file__).parents[1] / "examples" / "input" / "input_1.jpg"
IMAGE_SIZE = (100, 100)
NEAR = MAX_RIVAL_SIMILARITY + 0.01
FAR = MAX_RIVAL_SIMILARITY - 0.01


def test_select_objects_applies_every_rule():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # kept
            [0.0, 0.0, 10.0, 9.0],  # overlaps the first one with less objectness
            [50.0, 50.0, 60.0, 60.0],  # too little objectness
            [0.0, 20.0, 100.0, 100.0],  # covers most of the image
            [70.0, 100.0, 80.0, 100.0],  # empty, it was inside the padding
        ]
    )
    objectness = torch.tensor([0.9, 0.8, 0.1, 0.9, 0.9])
    assert select_objects(boxes, objectness, IMAGE_SIZE).tolist() == [0]


def test_select_objects_accepts_an_image_without_candidates():
    assert select_objects(torch.zeros((2, 4)), torch.zeros(2), IMAGE_SIZE).tolist() == []


def test_select_matches_rejects_a_box_that_a_rival_explains_better():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.9])
    assert select_matches(boxes, scores, torch.tensor([2.0, 0.5]), 0.2).tolist() == [0]


def test_select_matches_rejects_low_scores():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert select_matches(boxes, torch.tensor([0.1]), torch.tensor([2.0]), 0.2).tolist() == []


def test_words_of_the_query_are_not_rivals():
    rivals = find_rivals("cardboard box", torch.full((len(VOCABULARY),), FAR))
    assert not rivals[VOCABULARY.index("box")]
    assert rivals[VOCABULARY.index("bag")]


def test_synonyms_of_the_query_are_not_rivals():
    similarities = torch.full((len(VOCABULARY),), FAR)
    similarities[VOCABULARY.index("shoe")] = NEAR
    rivals = find_rivals("sneaker", similarities)
    assert not rivals[VOCABULARY.index("shoe")]
    assert rivals[VOCABULARY.index("bag")]


def test_display_name_hides_doubtful_words():
    assert display_name("shoe", MIN_NAME_SCORE) == "shoe"
    assert display_name("shoe", MIN_NAME_SCORE - 0.01) == UNKNOWN_NAME


@pytest.mark.parametrize("query", [[], "", ["  "]])
def test_empty_query_does_not_load_the_model(query):
    detector = ObjectDetector()
    assert detector.detect(None, query) == ()
    assert "_model" not in vars(detector)


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector()


@pytest.fixture(scope="module")
def features(detector):
    return detector.encode_image(load_image(IMAGE_PATH))


@pytest.mark.slow
def test_all_mode_finds_objects_with_names_and_scores(detector, features):
    objects = detector.detect(features, "ALL")
    assert objects
    assert all(item.box.area > 0 and 0 <= item.score <= 1 for item in objects)
    assert all(item.name in VOCABULARY or item.name == UNKNOWN_NAME for item in objects)


@pytest.mark.slow
def test_query_mode_returns_only_requested_names(detector, features):
    objects = detector.detect(features, ["Mug", "banana"])
    assert {item.name for item in objects} == {"mug"}


@pytest.mark.slow
def test_absent_object_gives_no_boxes(detector, features):
    assert detector.detect(features, "banana") == ()


@pytest.mark.slow
def test_name_outside_the_vocabulary_is_accepted(detector, features):
    assert detector.detect(features, "enamel mug")
