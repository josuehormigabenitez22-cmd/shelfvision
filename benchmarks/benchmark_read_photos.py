"""Read the codes in the example photos and compare them with the values of their labels.

Any value that is not on a label of the photo is a false read. For each code read, the outline the
reader gives is compared with the label's true corners: the normal and the owner of a code are taken
inside that outline, so an outline that spills over the background spoils both. The reader's
thresholds were tuned on other photos, so this measures how well they carry over to new scenes.

Run it from the project root with: pixi run python benchmarks/benchmark_read_photos.py
"""

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np
from ground_truth import LABELS, OBJECT_CODES

from shelfvision.barcodes import read_barcodes
from shelfvision.images import load_image

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
MIN_OUTLINE_IOU = 0.5


def read_values(path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Return the outline of each value read in an image, and the seconds read_barcodes took."""
    gray = cv2.cvtColor(np.asarray(load_image(path)), cv2.COLOR_RGB2GRAY)
    start = perf_counter()
    outlines = {barcode.value: np.float32(barcode.corners) for barcode in read_barcodes(gray)}
    return outlines, perf_counter() - start


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two convex quadrilaterals."""
    shared, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - shared
    return shared / union if union > 0 else 0.0


def listed(values: list[str]) -> str:
    return ", ".join(values) or "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    folder = parser.parse_args().images

    found = expected = false_reads = outlines_right = 0
    seconds: list[float] = []
    for image, codes in OBJECT_CODES.items():
        outlines, elapsed = read_values(folder / image)
        values = set(outlines)
        truth = {label["value"]: np.float32(label["corners"]) for label in LABELS[image]}
        overlaps = {value: polygon_iou(outlines[value], truth[value]) for value in values & set(truth)}
        loose = sorted(f"{value} ({overlap:.2f})" for value, overlap in overlaps.items()
                       if overlap < MIN_OUTLINE_IOU)
        outlines_right += len(overlaps) - len(loose)
        seconds.append(elapsed)
        missing = [code for code in codes if code not in values]
        false = sorted(values - set(codes))
        found += len(codes) - len(missing)
        expected += len(codes)
        false_reads += len(false)
        print(f"{image}: missing {listed(missing)}; false {listed(false)}; "
              f"loose outlines {listed(loose)}; {elapsed:.1f} s")

    print(f"Codes read {found}/{expected}, false reads {false_reads}. The goal for false reads is 0")
    print(f"Outlines that match their label (IoU >= {MIN_OUTLINE_IOU}): {outlines_right}/{found}")
    print(f"Seconds per image: mean {mean(seconds):.1f}, min {min(seconds):.1f}, max {max(seconds):.1f}")
    raise SystemExit(1 if false_reads else 0)


if __name__ == "__main__":
    main()
