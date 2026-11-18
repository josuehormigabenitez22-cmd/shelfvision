"""The orchestrator on a drawn image with two codes, with stand-ins for OWLv2 and MoGe."""

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from shelfvision.code128 import CodeSet, encode, symbols_to_widths
from shelfvision.datatypes import Box, DetectedObject, Outline
from shelfvision.normals import Intrinsics, SurfaceMap
from shelfvision.orchestrator import Orchestrator
from shelfvision.relations import NO_BARCODE, NO_MATCH
from shelfvision.synthetic import render_label

SIDE = 600
SHIFT = 90
# "Shoe" is printed on a box and "Mug" on a cup, so only geometry can link values and objects
ALL_MODE = (
    DetectedObject("box", Box(100, 150, 500, 270), 0.6),
    DetectedObject("object", Box(100, 330, 500, 450), 0.2),
    DetectedObject("scissors", Box(520, 20, 590, 120), 0.5),
)
QUERY_MODE = (
    DetectedObject("box", Box(95, 145, 505, 275), 0.5),
    DetectedObject("cup", Box(100, 330, 500, 450), 0.4),
    DetectedObject("scissors", Box(520, 20, 590, 120), 0.5),
)


class FakeDetector:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def encode_image(self, image):
        self.calls["encode"] += 1
        return "features"

    def detect_all(self, features):
        self.calls["all"] += 1
        return ALL_MODE

    def detect_names(self, features, names):
        self.calls["names"] += 1
        return tuple(item for item in QUERY_MODE if item.name in names)


class FakeEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, image):
        self.calls += 1
        width, height = image.size
        away = np.tile(np.float32([0.0, 0.0, 1.0]), (height, width, 1))
        return SurfaceMap(away, 1.0, Intrinsics(600.0, 600.0, width / 2, height / 2))


def label(text: str) -> np.ndarray:
    return render_label(symbols_to_widths(encode([(CodeSet.B, text)])), 3.0, 60, 0.0, SIDE)


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    gray = np.minimum(np.roll(label("Shoe"), -SHIFT, axis=0), np.roll(label("Mug"), SHIFT, axis=0))
    path = tmp_path_factory.mktemp("scene") / "two codes.png"
    Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB").save(path)
    return path


@pytest.fixture
def parts():
    return FakeDetector(), FakeEstimator()


@pytest.fixture
def scene(image_path, parts):
    detector, estimator = parts
    return Orchestrator(detector, estimator).open(image_path)


def test_detect_uses_the_query_mode_for_names_and_the_all_mode_for_all(scene):
    assert scene.detect(["box", "cup"]) == QUERY_MODE[:2]
    assert scene.detect("ALL") == ALL_MODE


def test_the_image_is_encoded_once_and_a_repeated_query_is_not_run_again(scene, parts):
    detector, _ = parts
    scene.detect(["cup", "box"])
    scene.detect(["box", "cup"])
    scene.detect(["cup", "box"])
    assert detector.calls == Counter(encode=1, names=2)


def test_codes_of_the_requested_objects_only_with_normals_facing_the_camera(scene):
    (code,) = scene.read_barcodes(["box"])
    assert code.value == "Shoe"
    assert code.normal == pytest.approx((0.0, 0.0, -1.0))
    assert [code.value for code in scene.read_barcodes("cup")] == ["Mug"]


def test_all_gives_every_code(scene):
    assert sorted(code.value for code in scene.read_barcodes("all")) == ["Mug", "Shoe"]


def test_moge_does_not_run_when_no_requested_object_has_a_code(scene, parts):
    _, estimator = parts
    assert scene.read_barcodes(["scissors"]) == ()
    assert scene.read_barcodes([]) == ()
    assert estimator.calls == 0


def test_task_3_answers_both_ways_and_explains_empty_answers(scene):
    assert scene.relate("Shoe").answers == ("box",)
    assert scene.relate("box").answers == ("Shoe",)
    assert scene.relate("cup").answers == ("Mug",)
    assert scene.relate("scissors").reason == NO_BARCODE
    assert scene.relate("bottle").reason == NO_MATCH


def test_task_3_needs_no_normal_map(scene, parts):
    _, estimator = parts
    scene.relate("Shoe")
    scene.relate("box")
    assert estimator.calls == 0


def test_output_images_add_the_prompt_and_timings_name_the_stages(scene, image_path):
    width, height = Image.open(image_path).size
    assert scene.draw_detections(["box"]).size[1] > height
    assert scene.draw_barcodes("all").size == scene.draw_detections("all").size
    assert set(scene.timings) == {
        "load image",
        "encode image (OWLv2)",
        "find all objects",
        "find requested objects",
        "read barcodes",
        "estimate normals (MoGe)",
    }
    assert all(seconds >= 0 for seconds in scene.timings.values())
    assert scene.draw_barcodes("all").size[0] == width


def test_a_missing_image_fails_only_when_used(parts, tmp_path):
    scene = Orchestrator(*parts).open(tmp_path / "missing.jpg")
    with pytest.raises(FileNotFoundError):
        scene.detect("all")


def test_duplicated_and_differently_written_names_are_one_query(scene, parts):
    detector, _ = parts
    assert scene.detect(["box", "BOX", "  box "]) == scene.detect("box") == QUERY_MODE[:1]
    assert detector.calls["names"] == 1


@pytest.mark.parametrize("query", [[], "", "   ", ["", " "]])
def test_empty_queries_give_nothing_and_no_error(scene, query):
    assert scene.detect(query) == ()
    assert scene.read_barcodes(query) == ()


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_task_3_query_gives_no_match(scene, query):
    assert scene.relate(query).reason == NO_MATCH


def test_all_in_a_list_and_in_any_case_selects_everything(scene):
    assert scene.detect(["mug", "All"]) == ALL_MODE
    assert len(scene.read_barcodes(["ALL"])) == 2


def test_a_query_that_is_not_text_is_rejected(scene):
    with pytest.raises(TypeError):
        scene.detect(["box", 3])


def test_report_holds_the_requested_objects_and_their_codes_with_owner_and_normal(scene):
    report = scene.report(["box"])
    assert report["query"] == ["box"]
    assert [item["name"] for item in report["objects"]] == ["box"]
    (code,) = report["barcodes"]
    assert code["value"] == "Shoe" and code["object"] == "box"
    assert code["normal"] == [0.0, 0.0, -1.0]


def test_report_for_all_names_each_code_after_its_all_mode_owner(scene):
    owners = {code["value"]: code["object"] for code in scene.report("all")["barcodes"]}
    assert owners == {"Shoe": "box", "Mug": "object"}


class FakeSegmenter:
    """Outlines each box with a thin strip at its left end, except the "object" box, whose outline
    reaches up over the box above it and so covers the "Shoe" label."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def encode_image(self, image):
        self.calls["encode"] += 1
        return "embedding"

    def outlines(self, embedding, boxes):
        self.calls["outlines"] += 1
        return tuple(TALL if box == ALL_MODE[1].box else strip(box) for box in boxes)


def strip(box: Box) -> Outline:
    return Outline((((box.x0, box.y0), (box.x0 + 10, box.y0), (box.x0 + 10, box.y1), (box.x0, box.y1)),))


TALL = Outline((((100.0, 140.0), (500.0, 140.0), (500.0, 450.0), (100.0, 450.0)),))


@pytest.fixture
def segmented(image_path, parts):
    detector, estimator = parts
    segmenter = FakeSegmenter()
    return Orchestrator(detector, estimator, segment=True, segmenter=segmenter).open(image_path), segmenter


def test_segmentation_is_off_by_default_and_a_segmenter_alone_does_not_turn_it_on(image_path, parts):
    detector, estimator = parts
    segmenter = FakeSegmenter()
    orchestrator = Orchestrator(detector, estimator, segmenter=segmenter)
    assert orchestrator.segmenter is None
    scene = orchestrator.open(image_path)
    assert all(item.outline is None for item in scene.detect("all"))
    assert scene.relate("Shoe").answers == ("box",)
    assert not segmenter.calls
    assert not any("SAM" in stage for stage in scene.timings)


def test_segmentation_outlines_every_object_and_encodes_the_image_once(segmented):
    scene, segmenter = segmented
    objects = scene.detect("all")
    assert [item.box for item in objects] == [item.box for item in ALL_MODE]
    assert all(item.outline is not None for item in objects)
    assert all(item.outline is not None for item in scene.detect(["box", "cup"]))
    scene.detect("all")
    assert segmenter.calls == Counter(encode=1, outlines=2)
    assert {"encode image (SAM)", "segment objects (SAM)"} <= set(scene.timings)
    assert "outline" in scene.report("all")["objects"][0]


def test_with_segmentation_task_3_follows_the_outlines(segmented):
    scene, _ = segmented
    assert scene.relate("Shoe").answers == ("object",)
    assert scene.relate("Mug").answers == ("object",)
