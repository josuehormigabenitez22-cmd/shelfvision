"""Link each barcode to the object it is printed on, and answer in both directions.

The link is purely geometric: a barcode belongs to the smallest detected box that holds its center,
or, when segmentation is on, to the smallest outline that holds it.
A barcode's value is never compared with an object's name, so a label reading "Shoe" stuck on a mug
belongs to the mug.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from shelfvision.datatypes import Barcode, DetectedObject, clean_name

# A query box and an "all" box overlapping this much are taken as the same object
SAME_OBJECT_IOU: Final = 0.5
NO_MATCH: Final = "no detected object or read barcode matches the query"
NO_BARCODE: Final = "the object was found, but no barcode was read on it"
NO_OWNER: Final = "the barcode was read, but its center is not inside any detected object"


@dataclass(frozen=True, slots=True)
class Relation:
    """Answer to a linking query: object names for a barcode value, or barcode values for a name
    The answers are empty only when the reason says why; a found answer has no reason
    """

    query: str
    answers: tuple[str, ...]
    reason: str | None = None


def owner(barcode: Barcode, objects: Sequence[DetectedObject]) -> DetectedObject | None:
    """Return the smallest object whose box holds the barcode's center, or None if no box does.

    With segmentation, an object whose outline holds the center comes first, the smallest if
    several do; the boxes decide only when no outline holds it. Without it, no object has an
    outline and only the boxes count.
    """
    center = barcode.center
    if outlined := [obj for obj in objects if obj.outline and obj.outline.contains(center)]:
        return min(outlined, key=lambda obj: obj.outline.area if obj.outline else 0.0)
    # The smallest box wins because a label on an object also lies inside any larger box around it
    holders = [obj for obj in objects if obj.box.contains(center)]
    return min(holders, key=lambda obj: obj.box.area, default=None)


def assign(
    barcodes: Sequence[Barcode], objects: Sequence[DetectedObject]
) -> list[tuple[Barcode, DetectedObject | None]]:
    """Pair every barcode with the object it is printed on, None when it is on no detected object"""
    return [(barcode, owner(barcode, objects)) for barcode in barcodes]


def merge_objects(
    requested: Sequence[DetectedObject], context: Sequence[DetectedObject]
) -> tuple[DetectedObject, ...]:
    """Return the requested objects, plus every context object that is not one of them.

    The requested objects come from a query, so they carry the names the user asked for; the
    context, from the "all" mode, keeps the other objects in play, so that a code printed on one
    of them is not handed to a requested object whose box happens to cover it.
    """
    others = (obj for obj in context if all(obj.box.iou(mine.box) < SAME_OBJECT_IOU for mine in requested))
    return (*requested, *others)


def _unique(texts: Iterable[str]) -> tuple[str, ...]:
    """Drop repeated texts and keep the first order."""
    return tuple(dict.fromkeys(texts))


def _found_or(query: str, answers: tuple[str, ...], reason: str) -> Relation:
    return Relation(query, answers) if answers else Relation(query, (), reason)


def relate(query: str, objects: Sequence[DetectedObject], barcodes: Sequence[Barcode]) -> Relation:
    """Answer a barcode value with the names of its objects, or a name with the values on its objects.

    Values are matched exactly first, because "Box" is a barcode value and "box" an object name.
    Anything else is taken as a name, compared without case or extra spaces. The values of an
    object come from left to right, and every object with that name contributes its barcodes.
    """
    pairs = assign(barcodes, objects)  # barcode -> object, worked out once
    # First see if the query is a barcode value; if not, it is an object name
    if holders := [holder for barcode, holder in pairs if barcode.value == query]:
        names = _unique(holder.name for holder in holders if holder is not None)
        return _found_or(query, names, NO_OWNER)
    name = clean_name(query)
    targets = [obj for obj in objects if clean_name(obj.name) == name]
    if not targets:
        return Relation(query, (), NO_MATCH)
    # Values printed on those objects, left to right
    on_targets = sorted(
        (barcode for barcode, holder in pairs if holder in targets), key=lambda barcode: barcode.center[0]
    )
    return _found_or(query, _unique(barcode.value for barcode in on_targets), NO_BARCODE)
