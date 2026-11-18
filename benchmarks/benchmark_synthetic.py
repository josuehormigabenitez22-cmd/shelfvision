"""Read random Code128 labels pasted on photo crops and count what read_barcodes gets right and wrong.

Each trial draws a text, module size, bar height, rotation, perspective and blur from the seed, and
pastes the label on a crop of an example photo. Neither the labels nor the crops were used to tune
the reader: the crops come from busy desks with keyboards, handwriting and wood grain. Seed 2027 was
never used for tuning.

Run it from the project root with: pixi run python benchmarks/benchmark_synthetic.py --seeds 2027
"""

import argparse
import os
import string
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter

import cv2
import numpy as np
from ground_truth import ALL_CODES

from shelfvision.barcodes import read_barcodes
from shelfvision.code128 import CodeSet, encode, symbols_to_widths
from shelfvision.images import load_image

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
TRIALS = 90
CROP = 1000
MODULES = (1.8, 2.2, 2.7, 3.5)
BLURS = (0.0, 0.7, 1.2)
PERSPECTIVES = (0.0, 0.03, 0.06)
BAR_MODULES = (20, 25, 30)
ALPHABETS = (
    string.ascii_letters,
    string.ascii_uppercase + string.digits,
    string.ascii_letters + string.digits + "-_./+ ",
)
SHORTEST, LONGEST = 3, 12
# Stripping the spaces of a text drawn only from spaces would leave nothing to encode
FALLBACK_TEXT = "ABC"
QUIET_MODULES = 10
# The label is drawn 8 times larger and warped at 4 times the crop, so that warping and blur act on
# sharp edges and narrow bars keep a realistic partial coverage once reduced
OVERSAMPLE = 8
WARP_SCALE = 4
PAPER, INK = 245, 20
NOISE = 3.0
LARGE_MODULE = 2.7
GOAL_TOTAL, GOAL_LARGE = 0.57, 0.71


@dataclass(frozen=True, slots=True)
class Trial:
    """Every random factor of one scene, drawn up front so that workers only render and read."""

    seed: int
    text: str
    module_px: float
    blur: float
    bar_modules: int
    perspective: float
    angle: float
    offsets: tuple[float, ...]
    background: np.ndarray
    noise_seed: int


@dataclass(frozen=True, slots=True)
class Outcome:
    read: bool
    wrong: tuple[str, ...]
    seconds: float


def code_widths(text: str) -> tuple[int, ...]:
    return symbols_to_widths(encode([(CodeSet.B, text)]))


def label_width(widths: Sequence[int], module_px: float) -> int:
    return int((sum(widths) + 2 * QUIET_MODULES) * module_px * OVERSAMPLE)


def draw_trials(seed: int, photos: Sequence[np.ndarray], count: int) -> Iterator[Trial]:
    """Draw each trial's factors in the prototype's order, so that a seed always gives the same scenes."""
    rng = Random(seed)
    for _ in range(count):
        module_px = rng.choice(MODULES)
        blur = rng.choice(BLURS)
        alphabet = rng.choice(ALPHABETS)
        letters = (rng.choice(alphabet) for _ in range(rng.randint(SHORTEST, LONGEST)))
        text = "".join(letters).strip() or FALLBACK_TEXT
        bar_modules = rng.choice(BAR_MODULES)
        photo = rng.choice(photos)
        left, top = rng.randrange(photo.shape[1] - CROP), rng.randrange(photo.shape[0] - CROP)
        angle = rng.uniform(0, 360)
        perspective = rng.choice(PERSPECTIVES)
        # Each corner moves by up to this many pixels along x and y
        jitter = perspective * label_width(code_widths(text), module_px) / OVERSAMPLE
        offsets = tuple(rng.uniform(-jitter, jitter) for _ in range(8))
        background = photo[top : top + CROP, left : left + CROP]
        yield Trial(seed, text, module_px, blur, bar_modules, perspective, angle, offsets, background,
                    rng.randrange(1 << 30))


def render_label(widths: Sequence[int], module_px: float, bar_modules: int) -> np.ndarray:
    """Return the label at OVERSAMPLE times its size: dark bars on paper, with a white margin around."""
    height = int(bar_modules * module_px * OVERSAMPLE)
    label = np.full((int(height * 1.25), label_width(widths, module_px)), PAPER, dtype=np.uint8)
    steps = [QUIET_MODULES * module_px * OVERSAMPLE, *(width * module_px * OVERSAMPLE for width in widths)]
    edges = np.cumsum(steps)
    for left, right in zip(edges[:-1:2], edges[1::2], strict=True):
        label[int(0.08 * height) : int(1.08 * height), int(left) : int(right)] = INK
    return label


def render_scene(trial: Trial) -> np.ndarray:
    """Warp the label onto its crop with the drawn perspective and rotation, then blur and add noise."""
    label = render_label(code_widths(trial.text), trial.module_px, trial.bar_modules)
    rows, cols = label.shape
    corners = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    target = np.float32(corners / OVERSAMPLE + np.reshape(trial.offsets, (4, 2)))
    theta = np.radians(trial.angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = (target - target.mean(axis=0)) @ rotation.T + CROP / 2
    to_crop = cv2.getPerspectiveTransform(corners, np.float32(target))
    to_large = np.diag([WARP_SCALE, WARP_SCALE, 1.0]) @ to_crop
    large = (CROP * WARP_SCALE, CROP * WARP_SCALE)
    warped = cv2.warpPerspective(label, to_large, large, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full_like(label, 255), to_large, large, flags=cv2.INTER_NEAREST)
    warped = cv2.resize(warped, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = cv2.resize(mask, (CROP, CROP), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    scene = trial.background.astype(np.float32) * (1 - mask) + warped * mask
    if trial.blur > 0:
        scene = cv2.GaussianBlur(scene, (0, 0), trial.blur)
    noise = np.random.default_rng(trial.noise_seed).normal(0.0, NOISE, scene.shape)
    return np.clip(scene + noise, 0, 255).astype(np.uint8)


def read_trial(trial: Trial) -> Outcome:
    scene = render_scene(trial)
    start = perf_counter()
    values = {barcode.value for barcode in read_barcodes(scene)}
    seconds = perf_counter() - start
    # A crop can show one of the photos' own labels; reading it is right, not a false read
    return Outcome(trial.text in values, tuple(sorted(values - {trial.text} - ALL_CODES)), seconds)


def use_one_thread() -> None:
    """Each worker reads one scene at a time, so OpenCV threads would only compete for the cores."""
    cv2.setNumThreads(1)


def share(outcomes: Sequence[Outcome]) -> float:
    return sum(outcome.read for outcome in outcomes) / len(outcomes) if outcomes else 0.0


def count(outcomes: Sequence[Outcome]) -> str:
    return f"{sum(outcome.read for outcome in outcomes)}/{len(outcomes)} ({share(outcomes):.0%})"


def print_breakdown(results: Sequence[tuple[Trial, Outcome]], factor: str) -> None:
    value_of = attrgetter(factor)
    for value in sorted({value_of(trial) for trial, _ in results}):
        group = [outcome for trial, outcome in results if value_of(trial) == value]
        print(f"  {factor} {value}: {count(group)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027])
    parser.add_argument("--trials", type=int, default=TRIALS, help="trials per seed")
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER, help="folder of the background photos")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = sorted(arguments.images.glob("*.jpg"))
    photos = [cv2.cvtColor(np.asarray(load_image(path)), cv2.COLOR_RGB2GRAY) for path in paths]
    trials = [trial for seed in arguments.seeds for trial in draw_trials(seed, photos, arguments.trials)]
    start = perf_counter()
    with ProcessPoolExecutor(arguments.workers, initializer=use_one_thread) as pool:
        results = list(zip(trials, pool.map(read_trial, trials), strict=True))
    wall = perf_counter() - start
    outcomes = [outcome for _, outcome in results]
    large = [outcome for trial, outcome in results if trial.module_px >= LARGE_MODULE]
    wrong = [(trial.text, outcome.wrong) for trial, outcome in results if outcome.wrong]

    print(f"Read {count(outcomes)}; module >= {LARGE_MODULE} px {count(large)}")
    for factor in ("seed", "module_px", "blur", "perspective"):
        print_breakdown(results, factor)
    seconds = [outcome.seconds for outcome in outcomes]
    print(f"Seconds per {CROP} px scene: median {median(seconds):.1f}, mean {mean(seconds):.1f}, "
          f"max {max(seconds):.1f}; {wall:.0f} s with {arguments.workers} workers")
    print(f"Wrong values: {len(wrong)}. The goal is 0")
    for text, values in wrong:
        print(f"  {text!r} read as {', '.join(map(repr, values))}")
    goals = {
        f"total >= {GOAL_TOTAL:.0%}": share(outcomes) >= GOAL_TOTAL,
        f"module >= {LARGE_MODULE} px >= {GOAL_LARGE:.0%}": share(large) >= GOAL_LARGE,
        "no wrong values": not wrong,
    }
    for goal, met in goals.items():
        print(f"{'met' if met else 'MISSED'}: {goal}")
    raise SystemExit(0 if all(goals.values()) else 1)


if __name__ == "__main__":
    main()
