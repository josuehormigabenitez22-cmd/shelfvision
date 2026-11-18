"""Check the barcode normals against the real geometry of the example photos.

Each label was stuck either on something that lies flat on the table (a notebook, a phone, a mouse pad)
or on something that stands upright (a mug, a screen, the front of a box). So within one photo, two
flat labels must be parallel (0°), and a flat label and an upright one perpendicular (90°); two upright
labels can face different ways, so they are not compared. Comparing labels with each other needs no
estimate of the table. With --output, it also saves each photo with its barcodes and normal arrows, to
review them by eye.

Run it from the project root with: pixi run python benchmarks/benchmark_normals.py [--output folder]
"""

import argparse
from itertools import combinations
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np
from ground_truth import GROUND_TRUTH

from shelfvision.barcodes import read_barcodes
from shelfvision.draw import render_barcodes
from shelfvision.images import load_image, save_image
from shelfvision.normals import NormalEstimator, arrow_tip, with_normals

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
FLAT, UPRIGHT = 0, 90
# The angle each label makes with the table in the real scene, in degrees
POSES = {
    "MG214": UPRIGHT, "MG380": UPRIGHT, "MG57": UPRIGHT, "CL903": FLAT,
    "PH118": FLAT, "NB775": FLAT, "X001NB42K7": FLAT, "LT63": FLAT,
    "LT509": FLAT, "TB246": FLAT, "BK831": FLAT, "NB320": FLAT,
    "MN725": UPRIGHT, "PD19": FLAT,
    "MG466": UPRIGHT, "LT902": UPRIGHT, "PP137": FLAT,
    "LT388": UPRIGHT, "PL71": UPRIGHT, "PD650": FLAT, "NB204": FLAT,
    "MN513": UPRIGHT, "BX847": UPRIGHT,
}
def angle(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two directions in degrees, from 0 to 90, whichever way each one points."""
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), 0.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    parser.add_argument("--output", type=Path, help="folder for the images with arrows")
    arguments = parser.parse_args()
    estimator = NormalEstimator()
    errors: dict[int, list[float]] = {FLAT: [], UPRIGHT: []}
    focals: list[float] = []
    seconds: list[float] = []
    for image in GROUND_TRUTH:
        picture = load_image(arguments.images / image)
        start = perf_counter()
        surface = estimator.estimate(picture)
        seconds.append(perf_counter() - start)
        focals.append(surface.intrinsics.fx)
        barcodes = with_normals(read_barcodes(cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2GRAY)), surface)
        normals = {code.value: np.asarray(code.normal) for code in barcodes
                   if code.normal is not None and code.value in POSES}
        measured = []
        for first, second in combinations(sorted(normals), 2):
            if POSES[first] == POSES[second] == UPRIGHT:
                continue
            ideal = abs(POSES[first] - POSES[second])
            value = angle(normals[first], normals[second])
            errors[ideal].append(abs(value - ideal))
            measured.append(f"{first}/{second} {value:.1f}° (ideal {ideal}°)")
        print(f"{image}: " + ("; ".join(measured) or "no pair to compare"))
        if arguments.output:
            tips = [arrow_tip(code, surface.intrinsics) for code in barcodes]
            save_image(render_barcodes(picture, barcodes, tips, "all"), arguments.output / image)

    print(f"\nMoGe: {mean(seconds):.1f} s per image (first one {seconds[0]:.1f} s, model loading included)")
    print(f"Focal: {min(focals):.0f} to {max(focals):.0f} px")
    for ideal, label in ((FLAT, "flat / flat"), (UPRIGHT, "flat / upright")):
        values = errors[ideal]
        if values:
            print(f"{label}: {len(values)} pairs, error to the ideal {ideal}°: "
                  f"median {np.median(values):.1f}°, max {max(values):.1f}°")


if __name__ == "__main__":
    main()
