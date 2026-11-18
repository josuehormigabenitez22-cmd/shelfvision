"""Read Code128 barcodes from pixels, with no barcode library"""

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from shelfvision.code128 import START_SETS, STOP_WIDTHS, SYMBOL_WIDTHS, symbols_to_text
from shelfvision.datatypes import Barcode, Point
from shelfvision.locator import Candidate, find_candidates

SYMBOL_MODULES: Final = 11
STOP_MODULES: Final = 13
SYMBOL_RUNS: Final = 6
STOP_RUNS: Final = 7
# Blur and ink spread make every bar a bit wider and every space a bit narrower, by the same amount
_SYMBOLS: Final = np.array(SYMBOL_WIDTHS, dtype=float)
_STOP: Final = np.array(STOP_WIDTHS, dtype=float)
SYMBOL_PAIRS: Final = _SYMBOLS[:, :-1] + _SYMBOLS[:, 1:]
STOP_PAIRS: Final = _STOP[:-1] + _STOP[1:]
START_SYMBOLS: Final = list(START_SETS)
# How far a symbol can be from its pattern and still count in modules, summed over its pairs
MAX_SYMBOL_ERROR: Final = 2.2
MAX_START_ERROR: Final = 1.5
MAX_STOP_ERROR: Final = 1.8
# The standard asks for 10 modules after the stop, photos and printing leave less
MIN_QUIET_MODULES: Final = 5.0
# No bar inside a code is wider than 4 modules, so a narrower margin followed by a wide dark area,
# such as the object around a label, also ends the code
WIDE_DARK_MODULES: Final = 8.0
# Shortest possible code ->  start + one data symbol + checksum + stop
MIN_RUNS: Final = 3 * SYMBOL_RUNS + STOP_RUNS
# Lines are enlarged before they are cut into runs, so that an edge can fall between two pixels
UPSAMPLE: Final = 4
SHARPEN_AMOUNT: Final = 1.5
SHARPEN_WINDOW: Final = 25
MIN_LOCAL_WINDOW: Final = 9
LOCAL_WINDOW_DIVISOR: Final = 12
# Wider blur windows go through the FFT, see smooth()
FFT_WINDOW: Final = 511
LOW_PERCENTILE: Final = 5
HIGH_PERCENTILE: Final = 95
# A normal photo uses at least half of the 0-255 range between its darkest and brightest 1 %.
# Dark or washed-out photos are stretched up to that range so the locator thresholds keep working,
# photos that already have enough contrast are not touched
MIN_INTENSITY_SPAN: Final = 128.0
SPAN_PERCENTILES: Final = (1, 99)


@dataclass(frozen=True, slots=True)
class ProfileRead:
    """A valid code read from one line, with its mean distance per symbol to the exact patterns"""
    text: str
    error: float


def symbol_errors(widths: np.ndarray) -> np.ndarray:
    """Return the distance from six measured widths to every symbol, in modules"""
    # Add each width to the next one and turn the result into modules
    pairs = (widths[:-1] + widths[1:]) / (widths.sum() / SYMBOL_MODULES)
    # Distance to all the Code128 symbols in one go
    return np.abs(pairs - SYMBOL_PAIRS).sum(axis=1)


def _is_stop(runs: np.ndarray, position: int, module: float) -> bool:
    """Tell whether a stop begins here: its widths match and a quiet zone, or the line end, follows"""
    widths = runs[position : position + STOP_RUNS]
    if len(widths) < STOP_RUNS:
        return False
    after = position + STOP_RUNS
    if after < len(runs) and runs[after] < MIN_QUIET_MODULES * module:
        beyond = runs[after + 1] if after + 1 < len(runs) else 0.0
        if beyond < WIDE_DARK_MODULES * module:
            return False
    pairs = (widths[:-1] + widths[1:]) / (widths.sum() / STOP_MODULES)
    return float(np.abs(pairs - STOP_PAIRS).sum()) < MAX_STOP_ERROR


def _read_from(runs: np.ndarray, start: int) -> ProfileRead | None:
    """Read symbols from a start symbol until a stop that closes a valid code"""
    # The start symbol is 11 modules wide, so it tells how big one module is
    module = runs[start : start + SYMBOL_RUNS].sum() / SYMBOL_MODULES
    symbols: list[int] = []
    errors: list[float] = []
    for position in range(start, len(runs) - SYMBOL_RUNS + 1, SYMBOL_RUNS):
        # Careful: some symbols followed by a bar look just like the stop, so the reading only ends
        # there if the checksum is also right
        if symbols and _is_stop(runs, position, module) and (text := symbols_to_text(symbols)):
            return ProfileRead(text, float(np.mean(errors)))
        distances = symbol_errors(runs[position : position + SYMBOL_RUNS])
        best = int(distances.argmin())  # closest symbol
        if distances[best] > MAX_SYMBOL_ERROR:  # too far from every symbol: not a code
            return None
        symbols.append(best)
        errors.append(float(distances[best]))
    return None


def read_runs(runs: np.ndarray) -> ProfileRead | None:
    """Return the first valid code in bar and space widths that begin with a bar"""
    # The start symbol could begin at any bar, so try them all
    for start in range(0, len(runs) - MIN_RUNS + 1, 2):
        if symbol_errors(runs[start : start + SYMBOL_RUNS])[START_SYMBOLS].min() <= MAX_START_ERROR:
            read = _read_from(runs, start)
            if read is not None:
                return read
    return None

def line_to_runs(line: np.ndarray, threshold: np.ndarray | float) -> np.ndarray | None:
    """Return the widths between the dark and light changes of a line, starting at a bar"""
    dark = line < threshold
    # Positions where the line goes from dark to light or back
    edges = np.flatnonzero(dark[1:] != dark[:-1]) + 1
    if len(edges) <= MIN_RUNS:
        return None
    first_bar = 0 if dark[edges[0]] else 1
    return np.diff(edges[first_bar:]).astype(float)


def smooth(line: np.ndarray, window: int) -> np.ndarray:
    """Gaussian blur of a line with an odd window, as cv2.GaussianBlur with its default border

    A direct blur costs the line's length times the window. Wide strips give lines of 20000 samples and
    windows of 1700, so past FFT_WINDOW the same kernel is applied through the FFT, about 25x faster on
    such lines; the two agree to within 0.001 gray levels.
    """
    if window <= FFT_WINDOW:
        return cv2.GaussianBlur(line.reshape(-1, 1), (1, window), 0).ravel()
    kernel = cv2.getGaussianKernel(window, 0).ravel()
    padded = np.pad(line.astype(np.float64), window // 2, mode="reflect")  # OpenCV's BORDER_REFLECT_101
    size = 1 << (len(padded) + window - 2).bit_length()
    full = np.fft.irfft(np.fft.rfft(padded, size) * np.fft.rfft(kernel, size), size)
    return full[window - 1 : len(padded)].astype(line.dtype)


def _thresholds(line: np.ndarray) -> Iterator[np.ndarray | float]:
    """Yield a threshold that follows uneven light and one halfway between dark and light"""
    window = max(MIN_LOCAL_WINDOW, len(line) // LOCAL_WINDOW_DIVISOR) | 1
    yield smooth(line, window)
    low, high = np.percentile(line, [LOW_PERCENTILE, HIGH_PERCENTILE])
    yield (low + high) / 2


def read_profile(profile: np.ndarray) -> Iterator[ProfileRead]:
    """Yield every reading of a gray line, trying it plain and sharpened, and in both directions"""
    if profile.size < MIN_RUNS:
        return
    pixels = profile.astype(np.float32).reshape(1, -1)
    # Make the line 4x longer, so an edge can land between two of the original pixels
    raw = cv2.resize(pixels, (profile.size * UPSAMPLE, 1), interpolation=cv2.INTER_CUBIC).ravel()
    # Blurred copy for the unsharp mask just below
    soft = cv2.GaussianBlur(raw.reshape(-1, 1), (1, SHARPEN_WINDOW), 0).ravel()
    for line in (raw, raw + SHARPEN_AMOUNT * (raw - soft)):
        for threshold in _thresholds(line):
            runs = line_to_runs(line, threshold)
            if runs is None:
                continue
            # An upside-down code reads backwards. The reversed widths still have to start with a bar
            for oriented in (runs, runs[::-1] if len(runs) % 2 else runs[-2::-1]):
                read = read_runs(oriented)
                if read is not None:
                    yield read


def best_read(readings: Iterable[ProfileRead]) -> ProfileRead | None:
    """Return the reading closest to the exact patterns, or None when there are no readings"""
    return min(readings, key=lambda reading: reading.error, default=None)


# Room added at each side of a candidate, as a fraction of its width. It holds the quiet zone that
# must follow the stop, and the rest of the code when the locator found only part of it
QUIET_PAD: Final = 0.35
MIN_STRIP_ROWS: Final = 6
# Perspective and curved labels leave the bars slanted after the turn: x moves by slope * y
MAX_SHEAR: Final = 0.4
SHEAR_STEPS: Final = 33


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def strip_size(candidate: Candidate) -> tuple[int, int]:
    """Columns and rows of the strip cut around a candidate"""
    return round(candidate.width * (1 + 2 * QUIET_PAD)), max(MIN_STRIP_ROWS, round(candidate.height))


def aligned_strip(gray: np.ndarray, candidate: Candidate, slope: float) -> np.ndarray:
    """Cut the candidate turned so its bars stand vertical, sheared by slope around its middle row
    Turn and shear go into one warp from the original image, so sheared rows never run out of pixels
    """
    columns, rows = strip_size(candidate)
    # Rotate so the bars stand vertical and the candidate sits in the middle of the strip
    turn = cv2.getRotationMatrix2D(candidate.center, candidate.angle, 1.0)
    turn[:, 2] += (columns / 2 - candidate.center[0], rows / 2 - candidate.center[1])
    # Then shear, for bars that still lean after rotating (perspective, curved labels)
    shear = np.array([[1.0, slope, -slope * rows / 2], [0.0, 1.0, 0.0]])
    matrix = (_homogeneous(shear) @ _homogeneous(turn))[:2]
    return cv2.warpAffine(
        gray, matrix, (columns, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def sharpness(strip: np.ndarray) -> float:
    """How crisp the mean row is: only bars aligned with the columns keep their edges after averaging"""
    return float(np.square(np.diff(strip.mean(axis=0))).sum())


def ranked_shears(gray: np.ndarray, candidate: Candidate) -> list[float]:
    """Return every slope tried around the expected slant, the one that best aligns the bars first"""
    slopes = (candidate.slant + np.linspace(-MAX_SHEAR, MAX_SHEAR, SHEAR_STEPS)).tolist()
    return sorted(slopes, key=lambda slope: -sharpness(aligned_strip(gray, candidate, slope)))


# A text needs this many votes, and this many times the votes of any rival, to be accepted.
# A single line can still read wrong; several blocks agreeing on a wrong text is rare
MIN_VOTES: Final = 3
VOTE_MARGIN: Final = 3
MIN_BLOCK_ROWS: Final = 3


@dataclass(frozen=True, slots=True)
class Stage:
    """One round of votes: the best shears to use, block heights as fractions of the strip, strictness"""
    shears: int
    heights: tuple[float, ...]
    max_error: float
    # Costly rounds only run where earlier rounds already found some reading
    needs_votes: bool = False


# Cheap rounds first; each runs only if the earlier ones left no clear winner
STAGES: Final = (
    # The best alignments settle most printed labels at once
    Stage(shears=2, heights=(1, 1 / 3), max_error=1.3),
    # Thin modules only read when many rows are averaged at the exact shear
    Stage(shears=SHEAR_STEPS, heights=(1,), max_error=1.3),
    Stage(shears=SHEAR_STEPS // 2, heights=(1 / 2, 1 / 3), max_error=1.3),
    # Fanned or curved bars only line up over a few rows at a time, so readings must fit better
    Stage(shears=8, heights=(1 / 8, 1 / 20), max_error=1.05, needs_votes=True),
)


def row_blocks(strip: np.ndarray, heights: Sequence[float]) -> Iterator[np.ndarray]:
    """Yield the strip cut into stacked blocks of each height, given as a fraction of its rows."""
    rows = strip.shape[0]
    for height in heights:
        size = min(rows, max(MIN_BLOCK_ROWS, round(height * rows)))
        for top in range(0, rows - size + 1, size):
            yield strip[top : top + size]


def winner(votes: Counter[str]) -> str | None:
    """Return the text that clearly beats every other reading, or None"""
    ranked = votes.most_common(2)
    if not ranked:
        return None
    (text, count), runner_up = ranked[0], ranked[1][1] if len(ranked) > 1 else 0
    return text if count >= MIN_VOTES and count >= VOTE_MARGIN * runner_up else None


def _vote(
    gray: np.ndarray, candidate: Candidate, slopes: Sequence[float], stage: Stage, votes: Counter[str]
) -> None:
    """Add one vote for each row block whose best reading fits the stage's strictnes"""
    for slope in slopes[: stage.shears]:
        for block in row_blocks(aligned_strip(gray, candidate, slope), stage.heights):
            reading = best_read(read_profile(block.mean(axis=0)))
            if reading is not None and reading.error <= stage.max_error:
                votes[reading.text] += 1


def read_candidate(gray: np.ndarray, candidate: Candidate, stages: Sequence[Stage] = STAGES) -> str | None:
    """Return the text of a candidate once enough row blocks agree on it, or None"""
    slopes = ranked_shears(gray, candidate)
    votes: Counter[str] = Counter()
    # Cheapest stage first; stop as soon as one text clearly wins
    for stage in stages:
        if stage.needs_votes and not votes:
            return None
        _vote(gray, candidate, slopes, stage, votes)
        if text := winner(votes):
            return text
    return None


def _corners(candidate: Candidate) -> tuple[Point, Point, Point, Point]:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = cv2.boxPoints(candidate.rotated_rect).tolist()
    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


def with_enough_contrast(gray: np.ndarray) -> np.ndarray:
    """Stretch a dark or flat image to MIN_INTENSITY_SPAN; leave any other image unchange"""
    # Range between the darkest 1 % and the brightest 1 % of the pixels
    low, high = np.percentile(gray, SPAN_PERCENTILES)
    span = float(high - low)
    if span >= MIN_INTENSITY_SPAN or span <= 0:
        return gray
    return (gray - low) * (MIN_INTENSITY_SPAN / span)


def distinct(barcodes: Iterable[Barcode]) -> list[Barcode]:
    """Drop each barcode that repeats the value of an overlapping one: two candidates on one label

    Of the copies, the one with the smallest box stays: a striped area of background that also holds the
    label gives the same value with an outline far larger than the label, and the normal and the owner
    of a code are taken inside its outline. The barcodes kept come out in their original order.
    """
    barcodes = list(barcodes)
    kept: list[Barcode] = []
    for barcode in sorted(barcodes, key=lambda code: code.box.area):  # stable: ties keep their order
        if not any(other.value == barcode.value and other.box.iou(barcode.box) > 0 for other in kept):
            kept.append(barcode)
    return [barcode for barcode in barcodes if any(barcode is other for other in kept)]


def read_barcodes(gray: np.ndarray) -> list[Barcode]:
    """Locate and read every code in a gray image; candidates that do not read are dropped"""
    gray = with_enough_contrast(np.asarray(gray, dtype=np.float32))  # fixes dark photos first
    # Try to read every candidate and throw away the ones that give nothing
    found = ((candidate, read_candidate(gray, candidate)) for candidate in find_candidates(gray))
    return distinct(Barcode(text, _corners(candidate)) for candidate, text in found if text is not None)
