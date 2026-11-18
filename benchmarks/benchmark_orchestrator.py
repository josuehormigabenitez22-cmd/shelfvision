"""Run detection, barcodes and linking end to end on the example photos, through the orchestrator.

For each photo it times the first round of queries, which pays every stage once, and the same round
again, which the cache must answer in under 0.3 s. It runs every photo twice with fresh scenes to
check that the outputs are identical, asks for the barcodes of each object alone to check that only
its codes come back, shows the two linking examples, and times OWLv2 and MoGe with 4 and 8 torch
threads. With --output it also saves the detection and barcode images.

Run it from the project root with: pixi run python benchmarks/benchmark_orchestrator.py [--output folder]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from ground_truth import CODE_OWNERS, GROUND_TRUTH

from shelfvision.images import load_image, save_image
from shelfvision.orchestrator import Orchestrator, Scene

IMAGES_FOLDER = Path(__file__).parents[1] / "examples" / "input"
NAMES = ("laptop", "mug")
# Two linking examples: an object name and a barcode value
EXAMPLES = ("mug", "LT388")
OBJECTS_WITH_CODES = tuple(dict.fromkeys(CODE_OWNERS.values()))
THREADS = (4, 8)
MAX_REPEATED_SECONDS = 0.3


@dataclass(frozen=True, slots=True)
class ImageRun:
    image: str
    first_seconds: float
    repeated_seconds: float
    outputs: tuple
    timings: dict[str, float]


def run_round(scene: Scene) -> tuple:
    """Every task once with a list and once with "all", and the two linking examples."""
    return (
        scene.detect(NAMES),
        scene.detect("all"),
        scene.read_barcodes(NAMES),
        scene.read_barcodes("all"),
        tuple(scene.relate(query) for query in EXAMPLES),
    )


def run_image(orchestrator: Orchestrator, path: Path) -> tuple[ImageRun, Scene]:
    scene = orchestrator.open(path)
    start = perf_counter()
    outputs = run_round(scene)
    middle = perf_counter()
    run_round(scene)
    end = perf_counter()
    return ImageRun(path.name, middle - start, end - middle, outputs, dict(scene.timings)), scene


def selection(scene: Scene) -> tuple[int, int, list[str]]:
    """Ask for the barcodes of each object alone; count its codes found and the codes of other objects."""
    everything = {code.value for code in scene.read_barcodes("all")}
    found = expected = 0
    leaks: list[str] = []
    for name in OBJECTS_WITH_CODES:
        values = {code.value for code in scene.read_barcodes([name])}
        mine = {value for value in everything if CODE_OWNERS.get(value) == name}
        found += len(values & mine)
        expected += len(mine)
        leaks += [f"{value} for {name!r}" for value in sorted(values - mine)]
    return found, expected, leaks


def save_outputs(scene: Scene, folder: Path) -> None:
    stem = scene.path.stem
    for query, tag in ((list(NAMES), "list"), ("all", "all")):
        save_image(scene.draw_detections(query), folder / f"{stem}_objects_{tag}.jpg")
        save_image(scene.draw_barcodes(query), folder / f"{stem}_barcodes_{tag}.jpg")


def time_threads(orchestrator: Orchestrator, path: Path) -> dict[int, tuple[float, float]]:
    """Seconds of OWLv2 and of MoGe on one image for each thread count, after a warm-up run."""
    image = load_image(path)
    default = torch.get_num_threads()
    orchestrator.detector.encode_image(image)
    orchestrator.estimator.estimate(image)
    seconds = {}
    for threads in THREADS:
        torch.set_num_threads(threads)
        start = perf_counter()
        orchestrator.detector.encode_image(image)
        middle = perf_counter()
        orchestrator.estimator.estimate(image)
        seconds[threads] = (middle - start, perf_counter() - middle)
    torch.set_num_threads(default)
    return seconds


def describe(run: ImageRun) -> str:
    named, everything, codes_of_names, codes, (by_name, by_value) = run.outputs
    return (
        f"first {run.first_seconds:.1f} s, repeated {1000 * run.repeated_seconds:.0f} ms; "
        f"{len(named)} boxes for {list(NAMES)}, {len(everything)} for 'all'; "
        f"codes {[code.value for code in codes_of_names]} for {list(NAMES)}, {len(codes)} for 'all'; "
        f"relate({EXAMPLES[0]!r}) = {by_name.answers}, relate({EXAMPLES[1]!r}) = {by_value.answers}"
    )


def print_timings(first_runs: list[ImageRun], second_runs: list[ImageRun]) -> float:
    """Print the first and repeated rounds and the stages; return the slowest repeated round."""
    firsts = [run.first_seconds for run in first_runs]
    again = mean(run.first_seconds for run in second_runs)
    repeated = max(run.repeated_seconds for run in first_runs + second_runs)
    print(f"\nFirst round: mean {mean(firsts):.1f} s, from {min(firsts):.1f} to {max(firsts):.1f} s "
          f"(the first image also loads the models); second run, mean {again:.1f} s")
    print(f"Repeated round: at most {1000 * repeated:.0f} ms; the limit is {1000 * MAX_REPEATED_SECONDS:.0f}")
    for stage in first_runs[0].timings:
        print(f"  {stage}: mean {mean(run.timings.get(stage, 0.0) for run in first_runs):.2f} s")
    return repeated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=IMAGES_FOLDER)
    parser.add_argument("--output", type=Path, help="folder for the detection and barcode images")
    arguments = parser.parse_args()
    paths = [arguments.images / image for image in GROUND_TRUTH]
    orchestrator = Orchestrator()
    first_runs, found, expected, leaks = [], 0, 0, []
    for path in paths:
        run, scene = run_image(orchestrator, path)
        first_runs.append(run)
        image_found, image_expected, image_leaks = selection(scene)
        found, expected, leaks = found + image_found, expected + image_expected, leaks + image_leaks
        if arguments.output:
            save_outputs(scene, arguments.output)
    second_runs = [run_image(orchestrator, path)[0] for path in paths]
    identical = [first.outputs == again.outputs for first, again in zip(first_runs, second_runs, strict=True)]
    for run, same in zip(first_runs, identical, strict=True):
        print(f"{run.image}: {describe(run)}; second run {'identical' if same else 'DIFFERENT'}")

    repeated = print_timings(first_runs, second_runs)
    print(f"Identical outputs in two runs: {sum(identical)}/{len(identical)}")
    print(f"Barcodes by name: {found}/{expected} codes of the requested object, "
          f"codes of others: {leaks or 0}")
    for threads, (owl, moge) in time_threads(orchestrator, paths[0]).items():
        print(f"{threads} torch threads: OWLv2 {owl:.1f} s, MoGe {moge:.1f} s")
    passed = repeated < MAX_REPEATED_SECONDS and all(identical) and not leaks
    print("Targets reached" if passed else "TARGETS NOT REACHED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
