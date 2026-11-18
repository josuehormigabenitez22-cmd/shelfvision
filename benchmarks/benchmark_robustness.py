"""Robustness of the whole system on altered photos, empty images and unusual names.

Photo cases: each test photo is rotated, shrunk, darkened, blurred and compressed, and the system
runs detection, barcodes and linking on it with "all". Boxes are checked against the hand-made ones
moved with the image, codes against the values printed on the objects. Other cases: a code cut by
the border, an object filling the frame, a plain image and a piece of table. Name cases: names that
are not in the photos, names outside the vocabulary, and naming without the word "table" in the
template. Any
exception is reported as a result, since no valid input should raise one.

Run it from the project root with: pixi run python benchmarks/benchmark_robustness.py
"""

import argparse
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from benchmark_detector import ABSENT_NAMES
from ground_truth import ALL_CODES, GROUND_TRUTH, OBJECT_CODES
from PIL import Image, ImageEnhance, ImageFilter

from shelfvision.barcodes import read_barcodes
from shelfvision.datatypes import Box, DetectedObject
from shelfvision.detector import (
    NAMING_TEMPLATES,
    VOCABULARY,
    ImageFeatures,
    ObjectDetector,
    display_name,
    select_objects,
)
from shelfvision.images import load_image
from shelfvision.orchestrator import Orchestrator

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
# Two photos with four codes each: one seen from above, one from the front
TEST_IMAGES = ("input_2.jpg", "input_6.jpg")
MIN_IOU = 0.5
DARK_FACTOR = 0.35
BLUR_FRACTION = 0.001
JPEG_QUALITY = 10
CLOSE_MARGIN = 0.1
EMPTY_SIZE = (960, 540)
# Names a user could type for the objects in the photos, none of them in the vocabulary
SYNONYMS = {
    "coffee mug": "mug",
    "enamel mug": "mug",
    "notebook": "book",
    "smartphone": "phone",
    "notebook computer": "laptop",
    "computer screen": "monitor",
    "mousepad": "mouse pad",
    "potted plant": "plant",
}
TEMPLATES = {"a {} on a table (current)": NAMING_TEMPLATES, "a photo of a {}": ("a photo of a {}",)}

type Truth = tuple[tuple[str, Box], ...]
type Variant = Callable[[Image.Image, Truth], tuple[Image.Image, Truth]]


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    image: Image.Image
    truth: Truth
    expected_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    case: str
    boxes: str
    names: str
    extras: int
    codes: str
    false_reads: tuple[str, ...]
    normals: str
    seconds: float
    error: str | None = None


def move(truth: Truth, point: Callable[[float, float], tuple[float, float]]) -> Truth:
    """Move every box with the image: map two opposite corners and take the box around them."""
    return tuple((name, Box.around([point(box.x0, box.y0), point(box.x1, box.y1)])) for name, box in truth)


def rotated_90(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
    """Turn a quarter counterclockwise, so the point (x, y) goes to (y, width - x)."""
    width = image.width
    return image.transpose(Image.Transpose.ROTATE_90), move(truth, lambda x, y: (y, width - x))


def rotated_180(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
    width, height = image.size
    return image.transpose(Image.Transpose.ROTATE_180), move(truth, lambda x, y: (width - x, height - y))


def shrunk_to(width: int) -> Variant:
    def shrink(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
        factor = width / image.width
        resized = image.resize((width, round(image.height * factor)), Image.Resampling.LANCZOS)
        return resized, move(truth, lambda x, y: (x * factor, y * factor))

    return shrink


def darker(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
    return ImageEnhance.Brightness(image).enhance(DARK_FACTOR), truth


def blurred(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
    return image.filter(ImageFilter.GaussianBlur(BLUR_FRACTION * image.width)), truth


def compressed(image: Image.Image, truth: Truth) -> tuple[Image.Image, Truth]:
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return Image.open(BytesIO(buffer.getvalue())).convert("RGB"), truth


VARIANTS: dict[str, Variant] = {
    "original": lambda image, truth: (image, truth),
    "rotated 90°": rotated_90,
    "rotated 180°": rotated_180,
    "shrunk to 1920 px": shrunk_to(1920),
    "shrunk to 640 px": shrunk_to(640),
    "dark": darker,
    "blurred": blurred,
    f"JPEG quality {JPEG_QUALITY}": compressed,
}


def hand_boxes(image: str) -> Truth:
    return tuple((name, Box(*box)) for name, box in GROUND_TRUTH[image])


def expected_codes(image: str) -> tuple[str, ...]:
    return OBJECT_CODES[image]


def crop(image: Image.Image, truth: Truth, window: Box) -> tuple[Image.Image, Truth]:
    """Cut the window out of the image, with the boxes that still fall inside it, clipped."""
    def clipped(box: Box) -> Box:
        return Box(max(box.x0, window.x0) - window.x0, max(box.y0, window.y0) - window.y0,
                   min(box.x1, window.x1) - window.x0, min(box.y1, window.y1) - window.y0)

    inside = tuple((name, clipped(box)) for name, box in truth if box.iou(window) > 0)
    return image.crop((round(window.x0), round(window.y0), round(window.x1), round(window.y1))), inside


def free_window(truth: Truth, image_size: tuple[int, int], size: tuple[int, int]) -> Box | None:
    """The first window of the given size that touches no hand-made box, scanning by quarter steps."""
    (width, height), (window_width, window_height) = image_size, size
    windows = (
        Box(left, top, left + window_width, top + window_height)
        for top in range(0, height - window_height, window_height // 4)
        for left in range(0, width - window_width, window_width // 4)
    )
    return next((window for window in windows if all(window.iou(box) == 0 for _, box in truth)), None)


def match(objects: Sequence[DetectedObject], truth: Truth) -> tuple[int, int, int]:
    """Hand-made boxes found (IoU >= 0.5, one detection each), those with the right name, and
    detections matching none."""
    found = named = 0
    free = list(objects)
    for name, box in truth:
        best = max(free, key=lambda item: item.box.iou(box), default=None)
        if best is not None and best.box.iou(box) >= MIN_IOU:
            free.remove(best)
            found += 1
            named += best.name == name
    extras = sum(all(item.box.iou(box) < MIN_IOU for _, box in truth) for item in objects)
    return found, named, extras


def run_case(orchestrator: Orchestrator, path: Path, case: Case) -> Outcome:
    """Run detection, barcodes and linking with "all" on the case's image, saved losslessly so nothing
    else changes."""
    case.image.save(path)
    start = perf_counter()
    try:
        scene = orchestrator.open(path)
        objects, codes = scene.detect("all"), scene.read_barcodes("all")
        scene.relate("mug")
    # Reporting any failure is the point of this benchmark, so no exception may stop it
    except Exception as error:
        return Outcome(case.name, "-", "-", 0, "-", (), "-", perf_counter() - start, repr(error))
    found, named, extras = match(objects, case.truth)
    values = {code.value for code in codes}
    read = sum(code in values for code in case.expected_codes)
    false = tuple(sorted(values - ALL_CODES))
    normals = sum(code.normal is not None for code in codes)
    seconds = perf_counter() - start
    return Outcome(case.name, f"{found}/{len(case.truth)}", f"{named}/{found}", extras,
                   f"{read}/{len(case.expected_codes)}", false, f"{normals}/{len(codes)}", seconds)


def gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def photo_cases(folder: Path) -> Iterator[Case]:
    for image in TEST_IMAGES:
        picture, truth = load_image(folder / image), hand_boxes(image)
        for label, variant in VARIANTS.items():
            altered, moved = variant(picture, truth)
            yield Case(f"{image} {label}", altered, moved, expected_codes(image))


def border_case(folder: Path, image: str = "input_2.jpg", value: str = "NB775") -> Case:
    """Cut the photo through the middle of a label: that code must not come out with a wrong value."""
    picture = load_image(folder / image)
    codes = read_barcodes(gray(picture))
    cut = next(code for code in codes if code.value == value).center[0]
    cropped, inside = crop(picture, hand_boxes(image), Box(0, 0, cut, picture.height))
    whole = tuple(code.value for code in codes if max(x for x, _ in code.corners) < cut)
    return Case(f"{image} cut through the {value} label", cropped, inside, whole)


def close_case(folder: Path, image: str = "input_5.jpg", name: str = "mug", value: str = "MG466") -> Case:
    """Crop around one object, so that it fills the frame beyond the "all" mode's size limit."""
    picture, truth = load_image(folder / image), hand_boxes(image)
    box = next(box for truth_name, box in truth if truth_name == name)
    margin_x, margin_y = CLOSE_MARGIN * (box.x1 - box.x0), CLOSE_MARGIN * (box.y1 - box.y0)
    window = Box(max(0, box.x0 - margin_x), max(0, box.y0 - margin_y),
                 min(picture.width, box.x1 + margin_x), min(picture.height, box.y1 + margin_y))
    cropped, inside = crop(picture, truth, window)
    return Case(f"{image} {name} filling the frame", cropped, inside, (value,))


def empty_cases(folder: Path, image: str = "input_5.jpg") -> Iterator[Case]:
    yield Case("plain gray image", Image.new("RGB", (1920, 1080), (128, 128, 128)), (), ())
    picture, truth = load_image(folder / image), hand_boxes(image)
    if window := free_window(truth, picture.size, EMPTY_SIZE):
        yield Case(f"{image} piece of table without objects", crop(picture, truth, window)[0], (), ())


def name_all(
    detector: ObjectDetector, features: ImageFeatures, templates: tuple[str, ...]
) -> list[DetectedObject]:
    """The detector's "all" mode, naming its boxes with other templates."""
    chosen = select_objects(features.boxes, features.objectness, features.image_size)
    best = detector.score(features, detector.encode_names(VOCABULARY, templates))[chosen].max(dim=1)
    rows = zip(chosen.tolist(), best.indices.tolist(), best.values.tolist(), strict=True)
    return [DetectedObject(display_name(VOCABULARY[word], score), Box(*features.boxes[index].tolist()), score)
            for index, word, score in rows]


def check_names(detector: ObjectDetector, folder: Path) -> None:
    """Absent names, names outside the vocabulary and other naming templates, on every photo."""
    absent, found, present, named = Counter(), Counter(), Counter(), Counter()
    for image in GROUND_TRUTH:
        features, truth = detector.encode_image(load_image(folder / image)), hand_boxes(image)
        absent.update(item.name for item in detector.detect_names(features, ABSENT_NAMES))
        answers = detector.detect_names(features, tuple(SYNONYMS))
        for synonym, target in SYNONYMS.items():
            targets = [box for name, box in truth if name == target]
            if targets:
                present[synonym] += 1
                found[synonym] += any(item.name == synonym and item.box.iou(box) >= MIN_IOU
                                      for item in answers for box in targets)
        for label, templates in TEMPLATES.items():
            named[label] += match(name_all(detector, features, templates), truth)[1]
    print(f"\nAbsent names with a box, over {len(GROUND_TRUTH)} photos: {dict(absent) or 'none'}")
    print("Names outside the vocabulary, photos where the right object is found:")
    for synonym, target in SYNONYMS.items():
        print(f"  {synonym!r} ({target}): {found[synonym]}/{present[synonym]}")
    total = sum(len(GROUND_TRUTH[image]) for image in GROUND_TRUTH)
    for label, right in named.items():
        print(f"Names right in the 'all' mode with {label!r}: {right}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    folder = parser.parse_args().images
    orchestrator = Orchestrator(threads=os.cpu_count())
    # Each case is built when its turn comes, so that only one altered photo is in memory at a time
    cases = chain(photo_cases(folder), (border_case(folder), close_case(folder)), empty_cases(folder))
    header = f"{'boxes':>6} {'named':>6} {'extra':>5} {'codes':>6} {'normals':>7} {'s':>6}"
    print(f"{'case':<46} {header}  false reads or error")
    errors = 0
    with TemporaryDirectory() as temporary:
        for index, case in enumerate(cases):
            row = run_case(orchestrator, Path(temporary) / f"case_{index}.png", case)
            errors += row.error is not None
            columns = f"{row.boxes:>6} {row.names:>6} {row.extras:>5} {row.codes:>6} {row.normals:>7}"
            problems = row.error or ", ".join(row.false_reads) or "-"
            print(f"{row.case:<46} {columns} {row.seconds:>6.1f}  {problems}", flush=True)
    check_names(orchestrator.detector, folder)
    print(f"Exceptions: {errors}. The goal is 0")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
