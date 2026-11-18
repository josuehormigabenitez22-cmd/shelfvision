"""Plain-data reports of a scene's results, to save as JSON and use outside Python"""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from shelfvision.datatypes import ALL_OBJECTS, Barcode, DetectedObject, Query

# A hundredth of a pixel is finer than any box or corner can be; four decimals keep unit vectors
# and scores exact to well below what the models can tell apart
PIXEL_DECIMALS: Final = 2
UNIT_DECIMALS: Final = 4
TIME_DECIMALS: Final = 3


def _rounded(values: Iterable[float], decimals: int) -> list[float]:
    return [round(float(value), decimals) for value in values]


def object_record(item: DetectedObject) -> dict[str, Any]:
    """Name, score and box; and with segmentation, the outline as a list of polygons"""
    box = item.box
    record: dict[str, Any] = {
        "name": item.name,
        "score": round(item.score, UNIT_DECIMALS),
        "box": _rounded((box.x0, box.y0, box.x1, box.y1), PIXEL_DECIMALS),
    }
    if item.outline is not None:
        record["outline"] = [
            [_rounded(point, PIXEL_DECIMALS) for point in polygon] for polygon in item.outline.polygons
        ]
    return record


def barcode_record(barcode: Barcode, owner: DetectedObject | None) -> dict[str, Any]:
    """The value, corners and unit normal of a code, and the name of the object it is printed on"""
    return {
        "value": barcode.value,
        "corners": [_rounded(corner, PIXEL_DECIMALS) for corner in barcode.corners],
        "normal": None if barcode.normal is None else _rounded(barcode.normal, UNIT_DECIMALS),
        "object": owner.name if owner else None,
    }


def build_report(
    image: str,
    size: tuple[int, int],
    query: Query,
    objects: Sequence[DetectedObject],
    barcodes: Sequence[tuple[Barcode, DetectedObject | None]],
    timings: Mapping[str, float],
) -> dict[str, Any]:
    """Detections and barcodes for one query, with the object that owns each code and the seconds per stage"""
    return {
        "image": image,
        "size": list(size),
        "query": ALL_OBJECTS if query == ALL_OBJECTS else list(query),
        "objects": [object_record(item) for item in objects],
        "barcodes": [barcode_record(barcode, owner) for barcode, owner in barcodes],
        "seconds_per_stage": {stage: round(seconds, TIME_DECIMALS) for stage, seconds in timings.items()},
    }


def save_report(report: Mapping[str, Any], path: Path) -> Path:
    """Write the report as indented UTF-8 JSON, creating its folder if needed, and return the path"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
