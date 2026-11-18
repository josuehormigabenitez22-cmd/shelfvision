"""Measure the barcode-object link on the example photos, with hand-made boxes and with the detector's.

Every code read is checked against the object it is printed on. With hand-made boxes the owner must
be that very object, not just one with the right name: input_1 holds eight mugs. With the boxes of
the detector's "all" mode, the owner's box must overlap the true box (IoU >= 0.5), and its name is
checked apart, since naming errors are the detector's. Then
relate() is asked both ways on the detector's boxes: each value read, and each object in the image.

Run it from the project root with: pixi run python benchmarks/benchmark_relations.py
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ground_truth import CODE_OWNERS, GROUND_TRUTH, owner_box

from shelfvision.barcodes import read_barcodes
from shelfvision.datatypes import Barcode, Box, DetectedObject
from shelfvision.detector import ObjectDetector
from shelfvision.images import load_image
from shelfvision.relations import owner, relate

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
MIN_IOU = 0.5
HAND_SCORE = 1.0


@dataclass(frozen=True, slots=True)
class CodeCheck:
    """Where one code read ends up, with each set of boxes, against the object it is printed on."""

    value: str
    truth: str
    hand_owner: str | None
    hand_right: bool
    detector_owner: str | None
    box_right: bool

    @property
    def name_right(self) -> bool:
        return self.box_right and self.detector_owner == self.truth


@dataclass(frozen=True, slots=True)
class QueryCheck:
    query: str
    is_value: bool
    answers: tuple[str, ...]
    expected: tuple[str, ...]

    @property
    def right(self) -> bool:
        return self.answers == self.expected


def hand_objects(image: str) -> tuple[DetectedObject, ...]:
    return tuple(DetectedObject(name, Box(*box), HAND_SCORE) for name, box in GROUND_TRUTH[image])


def name_of(found: DetectedObject | None) -> str | None:
    return found.name if found else None


def check_code(
    barcode: Barcode, image: str, hand: Sequence[DetectedObject], detected: Sequence[DetectedObject]
) -> CodeCheck:
    truth = CODE_OWNERS[barcode.value]
    true_box = Box(*owner_box(image, barcode.value))
    by_hand = owner(barcode, hand)
    hand_right = by_hand is not None and by_hand.box == true_box
    by_detector = owner(barcode, detected)
    box_right = by_detector is not None and by_detector.box.iou(true_box) >= MIN_IOU
    return CodeCheck(barcode.value, truth, name_of(by_hand), hand_right, name_of(by_detector), box_right)


def expected_values(name: str, barcodes: Sequence[Barcode]) -> tuple[str, ...]:
    """Values read on the object called name, from left to right, as relate() should give them."""
    on_object = sorted(
        (code for code in barcodes if CODE_OWNERS[code.value] == name), key=lambda code: code.center[0]
    )
    return tuple(dict.fromkeys(code.value for code in on_object))


def check_queries(
    image: str, detected: Sequence[DetectedObject], barcodes: Sequence[Barcode]
) -> list[QueryCheck]:
    """Ask relate() for every value read and for every object in the image."""
    values = dict.fromkeys(code.value for code in barcodes)
    by_value = [
        QueryCheck(value, True, relate(value, detected, barcodes).answers, (CODE_OWNERS[value],))
        for value in values
    ]
    by_name = [
        QueryCheck(name, False, relate(name, detected, barcodes).answers, expected_values(name, barcodes))
        for name in dict.fromkeys(name for name, _ in GROUND_TRUTH[image])
    ]
    return by_value + by_name


def describe_misses(codes: Sequence[CodeCheck], queries: Sequence[QueryCheck]) -> str:
    misses = [f"{code.value} on hand box {code.hand_owner}" for code in codes if not code.hand_right]
    misses += [
        f"{code.value} on {code.detector_owner}{'' if code.box_right else ' (wrong box)'}"
        for code in codes
        if not code.name_right
    ]
    misses += [f"relate({query.query!r}) = {query.answers}" for query in queries if not query.right]
    return "; ".join(misses) or "-"


def count(flags: Sequence[bool]) -> str:
    return f"{sum(flags)}/{len(flags)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    folder = parser.parse_args().images
    detector = ObjectDetector()
    codes: list[CodeCheck] = []
    queries: list[QueryCheck] = []
    false_reads = 0
    for image in GROUND_TRUTH:
        picture = load_image(folder / image)
        read = read_barcodes(cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2GRAY))
        barcodes = [barcode for barcode in read if barcode.value in CODE_OWNERS]
        false_reads += len(read) - len(barcodes)
        detected = detector.detect_all(detector.encode_image(picture))
        image_codes = [check_code(code, image, hand_objects(image), detected) for code in barcodes]
        image_queries = check_queries(image, detected, barcodes)
        codes += image_codes
        queries += image_queries
        print(f"{image}: {len(barcodes)} codes; misses: {describe_misses(image_codes, image_queries)}")

    by_value = [query.right for query in queries if query.is_value]
    by_name = [query.right for query in queries if not query.is_value]
    print(f"\nCodes read: {len(codes)}; false reads left out: {false_reads}")
    print(f"Owner with hand-made boxes: {count([code.hand_right for code in codes])}")
    print(f"Owner with detector boxes: right box {count([code.box_right for code in codes])}, "
          f"right box and name {count([code.name_right for code in codes])}")
    print(f"relate(value) on detector boxes: {count(by_value)}")
    print(f"relate(name) on detector boxes, every object in the images: {count(by_name)}")


if __name__ == "__main__":
    main()
