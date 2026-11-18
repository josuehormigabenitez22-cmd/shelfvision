"""Measure the optional segmentation on the example photos: links by outline, and its cost.

Each photo runs the "all" mode with segment=True. Every code read is linked to an object by the
outlines (the boxes only when no outline holds it) and checked as in benchmark_relations.py: the
owner's box must overlap the true box (IoU >= 0.5), and its name is checked apart. It also tells how
many codes no outline holds, and the seconds SAM adds to each photo.

Run it from the project root with: pixi run python benchmarks/benchmark_segmentation.py
With --output it also saves the detection images with the outlines filled in.
"""

import argparse
import os
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
from ground_truth import CODE_OWNERS, GROUND_TRUTH, owner_box

from shelfvision.barcodes import read_barcodes
from shelfvision.datatypes import Box
from shelfvision.images import save_image
from shelfvision.orchestrator import Orchestrator
from shelfvision.relations import owner

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
MIN_IOU = 0.5
SAM_STAGES = ("encode image (SAM)", "segment objects (SAM)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    parser.add_argument("--output", type=Path, help="folder for the detection images with outlines")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    orchestrator = Orchestrator(threads=args.threads, device=args.device, segment=True)
    right_box = right_name = outside = total = 0
    seconds: dict[str, list[float]] = {stage: [] for stage in SAM_STAGES}
    for image in GROUND_TRUTH:
        scene = orchestrator.open(args.images / image)
        objects = scene.detect("all")
        gray = cv2.cvtColor(np.asarray(scene.image), cv2.COLOR_RGB2GRAY)
        codes = [code for code in read_barcodes(gray) if code.value in CODE_OWNERS]
        misses = []
        for code in codes:
            truth = CODE_OWNERS[code.value]
            found = owner(code, objects)
            box_ok = found is not None and found.box.iou(Box(*owner_box(image, code.value))) >= MIN_IOU
            in_outline = any(item.outline and item.outline.contains(code.center) for item in objects)
            right_box += box_ok
            right_name += box_ok and found is not None and found.name == truth
            outside += not in_outline
            if not box_ok:
                misses.append(f"{code.value} on {found.name if found else None}")
        total += len(codes)
        for stage in SAM_STAGES:
            seconds[stage].append(scene.timings.get(stage, 0.0))
        unoutlined = sum(item.outline is None for item in objects)
        print(f"{image}: {len(objects)} objects ({unoutlined} without outline), {len(codes)} codes, "
              f"SAM {sum(scene.timings.get(stage, 0.0) for stage in SAM_STAGES):.1f} s; "
              f"wrong box: {', '.join(misses) or '-'}", flush=True)
        if args.output:
            save_image(scene.draw_detections("all"), args.output / f"segmented_{Path(image).stem}.jpg")

    print(f"\nOwner by outlines: right box {right_box}/{total}, right box and name {right_name}/{total}")
    print(f"Codes held by no outline (boxes decided): {outside}/{total}")
    for stage, values in seconds.items():
        print(f"{stage}: {mean(values):.2f} s per photo ({min(values):.2f} to {max(values):.2f})")


if __name__ == "__main__":
    main()
