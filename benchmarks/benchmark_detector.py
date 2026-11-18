"""Measure the detector against hand-estimated boxes, which are never used by the solution itself.

Run it from the project root with: pixi run python benchmarks/benchmark_detector.py
The photos show seven different desks; several hold more than one object with the same name.
"""

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from ground_truth import CLASSES, GROUND_TRUTH

from shelfvision.datatypes import Box, DetectedObject
from shelfvision.detector import UNKNOWN_NAME, ImageFeatures, ObjectDetector
from shelfvision.images import load_image

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
ABSENT_NAMES = (
    "banana", "hammer", "apple", "spoon", "shoe", "wrench", "bottle", "scissors", "umbrella", "drill",
)
QUERY_NAMES = (*CLASSES, *ABSENT_NAMES)
MIN_IOU = 0.5
THRESHOLDS = tuple(value / 100 for value in range(4, 70, 2))
# Share of the hand-drawn objects that each figure must reach for the run to pass
REQUIRED_BOX_SHARE = 0.75
REQUIRED_NAME_SHARE = 0.5
REQUIRED_QUERY_SHARE = 0.6

type Detections = dict[str, tuple[DetectedObject, ...]]


@dataclass(frozen=True, slots=True)
class Counts:
    # We initialize the case counter
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        """This function returns the count of cases"""
        return Counts(
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    @property
    def f1(self) -> float:
        """This formula is equivalent to the harmonic mean of Precision and Recall"""
        errors = self.false_positives + self.false_negatives
        return 2 * self.true_positives / (2 * self.true_positives + errors) if self.true_positives else 0.0


def count_query(detected: tuple[DetectedObject, ...], image: str, name: str, threshold: float) -> Counts:
    """Compare one name on one image: each hand-drawn object of that name is found by at most one box,
    and every box that finds none is a false positive"""
    kept = sorted((item for item in detected if item.name == name and item.score >= threshold),
                  key=lambda item: -item.score)
    truths = [Box(*box) for truth_name, box in GROUND_TRUTH[image] if truth_name == name]
    free = list(truths)
    matches = 0
    for item in kept:
        best = max(free, key=item.box.iou, default=None)
        if best is not None and item.box.iou(best) >= MIN_IOU:
            free.remove(best)
            matches += 1
    return Counts(matches, len(kept) - matches, len(free))


def count_images(detections: Detections, images: list[str], threshold: float) -> Counts:
    """accumulates the results (TP, FP, and FN)"""
    counts = (
        count_query(detections[image], image, name, threshold) for image in images for name in QUERY_NAMES
    )

    return sum(counts, Counts())


def cross_validate(detections: Detections) -> tuple[Counts, list[float]]:
    """Select the F1-optimal confidence threshold using two-fold cross-validation"""
    images = sorted(GROUND_TRUTH)
    folds = images[0::2], images[1::2]
    total, chosen = Counts(), []
    for train, test in (folds, folds[::-1]):
        threshold = max(THRESHOLDS, key=lambda value: count_images(detections, train, value).f1)
        total += count_images(detections, test, threshold)
        chosen.append(threshold)
    return total, chosen


def report_all_mode(detector: ObjectDetector, features: dict[str, ImageFeatures]) -> tuple[int, int]:
    boxes_found = names_right = unknown = extras = 0
    errors = []

    # Evaluate all images using the detector "all" mode
    for image, truth in GROUND_TRUTH.items():
        objects = list(detector.detect(features[image], "all"))
        matched = []
        # Each hand-drawn object takes the free box that overlaps it most
        for name, corners in truth:
            free = [item for item in objects if item not in matched]
            best = max(free, key=lambda item: item.box.iou(Box(*corners)), default=None)
            if best is None or best.box.iou(Box(*corners)) < MIN_IOU:
                errors.append(f"{image}: {name} has no box")
                continue
            # Count the matched box and evaluate its predicted label
            matched.append(best)
            boxes_found += 1
            names_right += best.name == name
            unknown += best.name == UNKNOWN_NAME
            if best.name != name:
                errors.append(f"{image}: {name} named {best.name} ({best.score:.2f})")
        extras += len(objects) - len(matched)
    total = sum(len(truth) for truth in GROUND_TRUTH.values())
    print(f'"all" mode: {boxes_found}/{total} boxes, {names_right} right names, '
          f'{unknown} "{UNKNOWN_NAME}", {extras} extra boxes')
    print(*errors, sep="\n")
    # Return the number of detected boxes and correctly classified names
    return boxes_found, names_right


def report_query_mode(detector: ObjectDetector, features: dict[str, ImageFeatures]) -> int:
    # Use the lowest confidence threshold to collect as many candidate detections as possible
    lowest = min(THRESHOLDS)
    # Run query-based detection on every image using the requested object names
    detections = {
        image: detector.detect_names(features[image], QUERY_NAMES, min_score=lowest) for image in GROUND_TRUTH
    }
    # Evaluate the detections across different confidence thresholds
    counts, chosen = cross_validate(detections)

    print(f"\nQuery mode, {len(CLASSES)} present names and {len(ABSENT_NAMES)} absent, cross-validated: "
          f"{counts.true_positives} TP, {counts.false_positives} FP, {counts.false_negatives} FN, "
          f"thresholds {chosen}")
    
    # Run detection again using the detector's default confidence threshold
    default = {image: detector.detect_names(features[image], QUERY_NAMES) for image in GROUND_TRUTH}
    # Identify detections corresponding to object classes known to be absent
    for image, detected in default.items():
        absent = sorted(item.name for item in detected if item.name in ABSENT_NAMES)
        print(f"{image}: absent names with a box at the default threshold: {absent or 'none'}")
    # Return the total number of true positive detections
    return counts.true_positives


def main() -> None:
    # Initialize the object detector and prepare storage for image features and encoding times
    detector = ObjectDetector()
    features, seconds = {}, []
    # Load and encode every image in the ground truth dataset
    for image in GROUND_TRUTH:
        picture = load_image(IMAGES_FOLDER / image)
        # Measure the time required to extract visual features from the image
        start = perf_counter()
        features[image] = detector.encode_image(picture)
        seconds.append(perf_counter() - start)
    # Report the average, minimum and maximum image encoding times
    print(f"Image encoding: mean {mean(seconds):.1f} s, from {min(seconds):.1f} to {max(seconds):.1f} s\n")
    
    # Evaluate object localization and classification in all mode
    boxes_found, names_right = report_all_mode(detector, features)
    # Evaluate name-based object detection in query mode
    true_positives = report_query_mode(detector, features)

    # Measure the execution time of a two-name query using cached text embeddings
    start = perf_counter()
    detector.detect(features[next(iter(GROUND_TRUTH))], ["laptop", "mug"])
    print(f"\nQuery of 2 names with the text cached: {perf_counter() - start:.3f} s")
    # Check whether all minimum performance requirements have been satisfied
    total = sum(len(truth) for truth in GROUND_TRUTH.values())
    passed = (
        boxes_found >= REQUIRED_BOX_SHARE * total
        and names_right >= REQUIRED_NAME_SHARE * total
        and true_positives >= REQUIRED_QUERY_SHARE * total
    )
    # Report the final evaluation result
    print("Targets reached" if passed else "TARGETS NOT REACHED")
    # Exit with status 0 on success or 1 if any performance target was missed
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
