"""Measure the line reader on synthetic barcodes: how many it reads and how many it reads wrong.

Run it from the project root with: pixi run python benchmarks/benchmark_profiles.py
"""

import string
from collections import Counter
from itertools import product
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np

from shelfvision.barcodes import best_read, read_profile
from shelfvision.code128 import CodeSet, encode, symbols_to_text, symbols_to_widths
from shelfvision.synthetic import render_profile

SEEDS = (2026, 9001)
LINES_PER_CASE = 50
MODULES = (1.25, 1.5, 1.8, 2.2, 2.7, 3.5, 5.0)
BLURS = (0.0, 0.7, 1.2, 1.6)
INKS = (-0.5, 0.0, 0.5)
NOISE = 3.0
LETTERS = string.ascii_letters + string.digits + "-"
ONLY_SET_B = 0.70
ONLY_SET_C = 0.15


def random_letters(rng: Random, shortest: int, longest: int) -> str:
    """Generate a random string of letters with a length within the given bounds"""
    return "".join(rng.choices(LETTERS, k=rng.randint(shortest, longest)))


def random_digits(rng: Random, fewest_pairs: int, most_pairs: int) -> str:
    """Generate a random string of digits with an even length based on the specified pair range"""
    return "".join(rng.choices(string.digits, k=2 * rng.randint(fewest_pairs, most_pairs)))


def random_code(rng: Random) -> tuple[int, ...]:
    """Generate a random Code 128 sequence using set B, set C, or both"""
    kind = rng.random()
    if kind < ONLY_SET_B:
        return encode([(CodeSet.B, random_letters(rng, 3, 14))])
    if kind < ONLY_SET_B + ONLY_SET_C:
        return encode([(CodeSet.C, random_digits(rng, 2, 7))])
    return encode([(CodeSet.B, random_letters(rng, 2, 5)), (CodeSet.C, random_digits(rng, 1, 4))])


def main() -> None:
    lines: Counter[tuple[str, float]] = Counter()
    read: Counter[tuple[str, float]] = Counter()
    wrong: Counter[tuple[str, float]] = Counter()
    mistakes: list[str] = []
    discarded = 0
    seconds: list[float] = []
    for seed in SEEDS:
        rng = Random(seed)
        noise_rng = np.random.default_rng(seed)
        for module, blur, ink in product(MODULES, BLURS, INKS):
            for _ in range(LINES_PER_CASE):
                symbols = random_code(rng)
                truth = symbols_to_text(symbols)
                profile = render_profile(symbols_to_widths(symbols), module, blur, ink, NOISE, noise_rng)
                profile = profile[::-1] if rng.random() < 0.5 else profile
                start = perf_counter()
                readings = list(read_profile(profile))
                best = best_read(readings)
                seconds.append(perf_counter() - start)
                discarded += any(reading.text != truth for reading in readings)
                is_wrong = best is not None and best.text != truth
                if is_wrong:
                    mistakes.append(f"{truth!r} read as {best.text!r}, module {module} px")
                for key in (("module px", module), ("blur px", blur), ("ink px", ink)):
                    lines[key] += 1
                    read[key] += best is not None and best.text == truth
                    wrong[key] += is_wrong

    for name, values in (("module px", MODULES), ("blur px", BLURS), ("ink px", INKS)):
        print(f"{name:>10} {'read':>11} {'wrong':>6}")
        for value in values:
            key = (name, value)
            print(f"{value:>10} {read[key]:>5}/{lines[key]:<5} {wrong[key]:>6}")
        print()

    print(f"{len(seconds)} lines, {sum(read[('blur px', blur)] for blur in BLURS)} read")
    print(f"Mean of {1000 * mean(seconds):.1f} ms per line")
    print(f"Lines with a wrong reading among their attempts: {discarded}")
    print(f"Wrong values after keeping the best reading: {len(mistakes)}. The goal is 0")
    for mistake in mistakes:
        print(f"  {mistake}")
    raise SystemExit(1 if mistakes else 0)


if __name__ == "__main__":
    main()
