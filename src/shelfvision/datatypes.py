"""Simple data types used across the pipeline, independent of the model"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

# Defines a constant to represent all objects in the pipeline
ALL_OBJECTS: Final = "all"

# Defines custom type names for points, vectors...
type Point = tuple[float, float]
type Vector3 = tuple[float, float, float]
type Query = Literal["all"] | tuple[str, ...]
type Polygon = tuple[Point, ...]


@dataclass(frozen=True, slots=True) # Frozen so a box cannot change after it is created
class Box:
    """Axis-aligned rectangle in pixels of the original image"""
    x0: float
    y0: float
    x1: float
    y1: float

    # We check that the box points are correct
    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"Box corners are out of order: {self}")

    @classmethod
    def around(cls, points: Sequence[Point]) -> "Box":
        """Creates the smallest box enclosing a set of points"""
        xs, ys = zip(*points, strict=True) # separate the x from the y
        return cls(min(xs), min(ys), max(xs), max(ys)) # search the extremes and return it

    @property
    def area(self) -> float:
        """Calculate the box area"""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def contains(self, point: Point) -> bool:
        """Tell whether a point lies inside the box, borders included"""
        x, y = point
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def iou(self, other: "Box") -> float:
        """Return the intersection over union, 0 when the boxes have no common area"""
        width = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        height = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


def _polygon_area(polygon: Polygon) -> float:
    """Shoelace formula: the area of a simple polygon, whatever the order of its corners"""
    edges = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in edges)) / 2


def _polygon_contains(polygon: Polygon, point: Point) -> bool:
    """Even-odd rule: a point is inside when a ray from it crosses the edges an odd number of times"""
    x, y = point
    inside = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


@dataclass(frozen=True, slots=True)
class Outline:
    """Silhouette of an object: one or more outer polygons in pixels of the original image
    Holes are not kept, so the inside of a mug's handle counts as part of the mug.
    """

    polygons: tuple[Polygon, ...]

    @property
    def area(self) -> float:
        return sum(_polygon_area(polygon) for polygon in self.polygons)

    def contains(self, point: Point) -> bool:
        return any(_polygon_contains(polygon, point) for polygon in self.polygons)


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """An object found in an image, with its model score in the range 0 to 1
    The outline is there only when segmentation is turned on; otherwise the box is all there is
    """
    name: str
    box: Box
    score: float
    outline: Outline | None = None


@dataclass(frozen=True, slots=True)
class Barcode:
    """A decoded code: its value, its four corners and, once computed, its unit normal"""
    value: str
    corners: tuple[Point, Point, Point, Point]
    normal: Vector3 | None = None

    @property
    def box(self) -> Box:
        return Box.around(self.corners)

    @property
    def center(self) -> Point:
        xs, ys = zip(*self.corners, strict=True)
        return sum(xs) / len(xs), sum(ys) / len(ys) # mean of the four corners


def clean_name(name: str) -> str:
    """Function to clean the name"""
    return " ".join(name.casefold().split()) # casefold() is the lower() but useful for unicodes


def normalize_query(query: str | Iterable[str]) -> Query:
    """Return "all", or the requested names cleaned, lowercase and without duplicates"""
    # A single string is one name, not a list of letters
    raw_names = (query,) if isinstance(query, str) else tuple(query)
    if invalid := [name for name in raw_names if not isinstance(name, str)]:
        raise TypeError(f"Object names must be strings, got: {invalid!r}")
    # Clean every name, drop the empty ones and remove duplicates, keeping the order
    names = tuple(dict.fromkeys(filter(None, map(clean_name, raw_names))))
    return ALL_OBJECTS if ALL_OBJECTS in names else names
