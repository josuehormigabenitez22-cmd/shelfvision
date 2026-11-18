"""JSON reports built from hand-made results: no model and no image."""

import json

from shelfvision.datatypes import Barcode, Box, DetectedObject
from shelfvision.export import barcode_record, build_report, object_record, save_report

BOX = DetectedObject("box", Box(10.123, 20.456, 110.0, 80.0), 0.612345)
CODE = Barcode("7Q-41", ((30.0, 40.0), (70.0, 40.0), (70.0, 50.0), (30.0, 50.0)), (0.1, -0.5, -0.86023))


def test_object_record_rounds_pixels_and_scores():
    assert object_record(BOX) == {"name": "box", "score": 0.6123, "box": [10.12, 20.46, 110.0, 80.0]}


def test_barcode_record_keeps_corners_normal_and_owner():
    record = barcode_record(CODE, BOX)
    assert record == {
        "value": "7Q-41",
        "corners": [[30.0, 40.0], [70.0, 40.0], [70.0, 50.0], [30.0, 50.0]],
        "normal": [0.1, -0.5, -0.8602],
        "object": "box",
    }


def test_a_code_without_normal_or_owner_says_so():
    record = barcode_record(Barcode("A1", CODE.corners), None)
    assert record["normal"] is None and record["object"] is None


def test_report_shows_the_query_as_given_and_every_part():
    timings = {"read barcodes": 9.12345}
    report = build_report("input_1.jpg", (3840, 2160), ("box",), [BOX], [(CODE, BOX)], timings)
    assert report["query"] == ["box"]
    assert build_report("a.jpg", (1, 1), "all", [], [], {})["query"] == "all"
    assert report["size"] == [3840, 2160]
    assert [item["name"] for item in report["objects"]] == ["box"]
    assert [code["object"] for code in report["barcodes"]] == ["box"]
    assert report["seconds_per_stage"] == {"read barcodes": 9.123}


def test_saved_report_reads_back_the_same_even_with_spaces_and_accents(tmp_path):
    report = build_report("café 1.jpg", (10, 10), ("caja",), [BOX], [(CODE, BOX)], {})
    path = save_report(report, tmp_path / "salidas con espacio" / "café 1.json")
    assert json.loads(path.read_text(encoding="utf-8")) == report
