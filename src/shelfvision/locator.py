"""Find where Code128 barcodes may be: center, scan direction and extent, with no barcode library"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from shelfvision.datatypes import Point

# Bars give strong gradients that all point the same way: high energy and high coherence
TENSOR_WINDOW: Final = 21
MIN_COHERENCE: Final = 0.6
ENERGY_PERCENTILE: Final = 95
# When codes fill more than 5 % of the image the percentile lands inside them and cuts them up,
# so a fraction of the strongest gradients is also accepted as strong enough
ENERGY_FRACTION: Final = 0.2
TOP_PERCENTILE: Final = 99.9
OPEN_KERNEL: Final = 5
CLOSE_KERNEL: Final = 19
MIN_REGION_AREA: Final = 2000
# A crop around a region or a candidate, as a multiple of its size, leaves room for the whole code
CROP_SCALE: Final = 1.6
# Smallest patch, in pixels, that still measures a direction, and its size relative to the code
MIN_ANGLE_PATCH: Final = 20
ANGLE_PATCH: Final = 0.55
NO_ENERGY: Final = 1e-3


@dataclass(frozen=True, slots=True)
class Candidate:
    """A place that may hold a code. Turning the image by angle degrees makes its bars vertical"""
    center: Point
    angle: float
    width: float
    height: float
    # Expected slope of the bars in a strip cut at this angle, when they lean against the code's axis
    slant: float = 0.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def rotated_rect(self) -> tuple[Point, tuple[float, float], float]:
        """The candidate as OpenCV describes a rotated rectangle; width runs across the bars"""
        return self.center, (self.width, self.height), self.angle


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


def _dominant_angle(gx: np.ndarray, gy: np.ndarray) -> float:
    """Direction of the strongest gradients in degrees: across the bars, so along the scan line"""
    return float(0.5 * np.degrees(np.arctan2(2 * (gx * gy).sum(), (gx * gx - gy * gy).sum())))


def _regions(gray: np.ndarray) -> Iterator[tuple[Point, float, float]]:
    """Yield the center, size and angle of each area with strong gradients that share one direction"""
    gx, gy = _gradients(gray)
    # Structure tensor. In plain words: the energy says how strong the edges are around each pixel,
    # and the coherence says how much they all point the same way. Bars have a lot of both,
    # text and textures usually don't
    window = (TENSOR_WINDOW, TENSOR_WINDOW)
    xx, yy, xy = (cv2.blur(product, window) for product in (gx * gx, gy * gy, gx * gy))
    energy = xx + yy
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy**2) / (energy + NO_ENERGY)
    common, top = np.percentile(energy, [ENERGY_PERCENTILE, TOP_PERCENTILE])
    strong = energy > min(common, ENERGY_FRACTION * top)
    # Strong edges all in the same direction are probably bars. The opening removes small specks
    # and the closing fills the gaps between the bars
    mask = ((coherence > MIN_COHERENCE) & strong).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Every blob big enough is a region, with the angle of its own gradients
    for contour in contours:
        if cv2.contourArea(contour) < MIN_REGION_AREA:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        inside = np.zeros((height, width), np.uint8)
        cv2.drawContours(inside, [contour - [x, y]], -1, 1, thickness=-1)
        window = np.s_[y : y + height, x : x + width]
        selected = inside.astype(bool)
        angle = _dominant_angle(gx[window][selected], gy[window][selected])
        center, _, _ = cv2.minAreaRect(contour)
        yield center, float(max(width, height)), angle


def rectify(gray: np.ndarray, center: Point, angle: float, side: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a square crop turned so the bars stand vertical, and the transform back to the image"""
    side = max(round(side), 1)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[:, 2] += (side / 2 - center[0], side / 2 - center[1])  # so the region ends up in the middle
    crop = cv2.warpAffine(gray, matrix, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return crop, cv2.invertAffineTransform(matrix)


def local_angle(gray: np.ndarray, center: Point, half: int) -> float | None:
    """Scan direction measured on the code itself, or None when the center is outside the image"""
    x, y = round(center[0]), round(center[1])
    if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
        return None
    patch = gray[max(0, y - half) : y + half, max(0, x - half) : x + half]
    if min(patch.shape) < 3:
        return None
    return _dominant_angle(*_gradients(patch))


# Edges along one row of a turned crop. A code has many close edges and ends at its quiet zones
ROW_STEP: Final = 3
# A start symbol, one data symbol, the checksum and the stop already have 25 edges
MIN_EDGES: Final = 20
MIN_EDGE_STEP: Final = 5.0
EDGE_FRACTION: Final = 0.25
EDGE_PERCENTILE: Final = 98
# A gap this many times the usual one ends the code. 3.5 reads more synthetic labels but cuts
# two labels of the photos in half
GAP_FACTOR: Final = 6.0
MIN_GAP_LIMIT: Final = 14.0
BOX_CLOSE_KERNEL: Final = (9, 3)
MIN_BOX_WIDTH: Final = 50
MIN_BOX_HEIGHT: Final = 9
# Printed labels are high contrast; stitching and wood grain are not
MIN_CONTRAST: Final = 70
CONTRAST_PERCENTILES: Final = (5, 95)
# Two candidates that share more than this fraction of their union are the same code
MAX_OVERLAP: Final = 0.5
# Perspective or a curved surface can lean the bars against the code's own axis. Rows cut across
# the bars then leave the code, so past this tilt the rows follow the axis and the strip is sheared
MIN_TILT: Final = 8.0
MAX_TILT: Final = 40.0
# Only a clearly elongated striped area has a direction worth trusting
MIN_ELONGATION: Final = 2.0
# The first direction comes from a piece of the code, so it is measured again on the larger piece
TILT_PASSES: Final = 2

type Rect = tuple[int, int, int, int]


def _row_segments(profile: np.ndarray) -> list[tuple[int, int]]:
    """Return the stretches of a row with many close edges, split where a gap is too long"""
    # An edge is a local peak in the difference between neighbouring pixels
    steps = np.abs(np.diff(profile))
    threshold = max(MIN_EDGE_STEP, EDGE_FRACTION * np.percentile(steps, EDGE_PERCENTILE))
    middle = steps[1:-1]
    peaks = np.flatnonzero((middle > threshold) & (middle >= steps[:-2]) & (middle > steps[2:])) + 1
    if len(peaks) < MIN_EDGES:
        return []
    gaps = np.diff(peaks)
    # A gap much longer than usual means the code is over (or another one starts)
    cuts = np.flatnonzero(gaps > max(MIN_GAP_LIMIT, GAP_FACTOR * np.median(gaps)))
    firsts, lasts = np.r_[0, cuts + 1], np.r_[cuts, len(peaks) - 1]
    return [(peaks[a], peaks[b]) for a, b in zip(firsts, lasts, strict=True) if b - a + 1 >= MIN_EDGES]


def _striped_areas(crop: np.ndarray) -> list[np.ndarray]:
    """Join the striped stretches of every few rows into areas, returned as contours"""
    mask = np.zeros(crop.shape, np.uint8)
    for row in range(1, crop.shape[0] - 1, ROW_STEP):
        for left, right in _row_segments(crop[row - 1 : row + 2].mean(axis=0)):
            mask[row : row + ROW_STEP, left : right + 1] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(BOX_CLOSE_KERNEL, np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [area for area in contours if _is_large_enough(cv2.boundingRect(area))]


def _is_large_enough(box: Rect) -> bool:
    return box[2] >= MIN_BOX_WIDTH and box[3] >= MIN_BOX_HEIGHT


def _boxes(crop: np.ndarray) -> list[Rect]:
    """Bounding boxes (x, y, width, height) of the striped areas"""
    return [cv2.boundingRect(contour) for contour in _striped_areas(crop)]


def _tilt(area: np.ndarray) -> float:
    """Direction of an elongated area against the crop rows, in degrees; 0 when it has none"""
    moments = cv2.moments(area)
    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    spread = np.hypot(mu20 - mu02, 2 * mu11)
    major, minor = mu20 + mu02 + spread, mu20 + mu02 - spread
    if minor <= 0 or major / minor < MIN_ELONGATION**2:
        return 0.0
    return float(0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02)))


def _has_contrast(crop: np.ndarray, box: Rect) -> bool:
    x, y, width, height = box
    low, high = np.percentile(crop[y : y + height, x : x + width], CONTRAST_PERCENTILES)
    return high - low >= MIN_CONTRAST


def _box_center(inverse: np.ndarray, box: Rect) -> Point:
    x, y, width, height = box
    image_x, image_y = inverse @ np.array([x + width / 2, y + height / 2, 1.0])
    return float(image_x), float(image_y)


def _refine(gray: np.ndarray, center: Point, width: float) -> Candidate | None:
    """Measure the angle on the code alone and take the box that holds the center at that angle"""
    angle = local_angle(gray, center, max(MIN_ANGLE_PATCH, round(width * ANGLE_PATCH)))
    if angle is None:
        return None
    crop, inverse = rectify(gray, center, angle, CROP_SCALE * width)
    area = _central_area(crop)
    if area is None:
        return None
    bars_angle = angle
    # If the striped area is too tilted, follow the code's axis instead
    for _ in range(TILT_PASSES):
        tilt = _tilt(area)
        if not MIN_TILT <= abs(tilt) <= MAX_TILT:
            break
        # Scan along the code's axis instead; the bars then lean inside the strip
        length = max(cv2.minAreaRect(area)[1])
        center, angle = _box_center(inverse, cv2.boundingRect(area)), angle + tilt
        crop, inverse = rectify(gray, center, angle, CROP_SCALE * length)
        area = _central_area(crop)
        if area is None:
            return None
    _, _, box_width, box_height = box = cv2.boundingRect(area)
    slant = -float(np.tan(np.radians(angle - bars_angle)))
    return Candidate(_box_center(inverse, box), angle, float(box_width), float(box_height), slant)


def _central_area(crop: np.ndarray) -> np.ndarray | None:
    """The largest striped area whose bounding box holds the crop's center"""
    middle = crop.shape[0] / 2
    holding = [area for area in _striped_areas(crop) if _holds(cv2.boundingRect(area), middle)]
    return max(holding, key=lambda area: np.prod(cv2.boundingRect(area)[2:]), default=None)


def _holds(box: Rect, middle: float) -> bool:
    x, y, width, height = box
    return x <= middle <= x + width and y <= middle <= y + height


def _overlap(first: Candidate, second: Candidate) -> float:
    """Intersection over union of two rotated candidates"""
    kind, points = cv2.rotatedRectangleIntersection(first.rotated_rect, second.rotated_rect)
    if kind == cv2.INTERSECT_NONE or points is None:
        return 0.0
    shared = cv2.contourArea(points)
    return shared / (first.area + second.area - shared)


def find_candidates(gray: np.ndarray) -> list[Candidate]:
    """Return the places that may hold a code, largest first, each with its own angle and extent"""
    if gray.ndim != 2:
        raise ValueError(f"Expected a gray image with 2 dimensions, got shape {gray.shape}")
    gray = gray.astype(np.float32)
    found = []
    # Find the striped boxes of each region, then refine each one on the code itself
    for center, size, angle in _regions(gray):
        crop, inverse = rectify(gray, center, angle, CROP_SCALE * size)
        for box in _boxes(crop):
            if _has_contrast(crop, box) and (candidate := _refine(gray, _box_center(inverse, box), box[2])):
                found.append(candidate)
    # Overlap, not distance, decides duplicates, so a wide false candidate cannot hide a nearby code
    kept: list[Candidate] = []
    for candidate in sorted(found, key=lambda c: -c.area):
        if all(_overlap(candidate, other) <= MAX_OVERLAP for other in kept):
            kept.append(candidate)
    return kept
