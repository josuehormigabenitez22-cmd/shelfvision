"""Links between barcodes and objects, on hand-made boxes and barcodes: no model and no image."""

import pytest

from shelfvision.datatypes import Barcode, Box, DetectedObject
from shelfvision.relations import (
    NO_BARCODE,
    NO_MATCH,
    NO_OWNER,
    Relation,
    assign,
    merge_objects,
    owner,
    relate,
)


def code(value: str, x: float, y: float, half: float = 10.0) -> Barcode:
    """A barcode centered at (x, y)."""
    corners = (x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)
    return Barcode(value, corners)


def thing(name: str, x0: float, y0: float, x1: float, y1: float) -> DetectedObject:
    return DetectedObject(name, Box(x0, y0, x1, y1), 0.9)


TABLE = thing("table", 0, 0, 1000, 1000)
BOX = thing("box", 100, 100, 400, 300)
MUG = thing("mug", 500, 100, 700, 300)
SCISSORS = thing("scissors", 100, 600, 300, 800)
OBJECTS = (TABLE, BOX, MUG, SCISSORS)
# The values have nothing to do with the names, so only geometry can link them
ON_BOX_RIGHT = code("7Q-41", 350, 200)
ON_BOX_LEFT = code("Kz9", 150, 200)
ON_MUG = code("Shoe", 600, 200)
ON_NOTHING = code("LOOSE", 1500, 1500)
BARCODES = (ON_BOX_RIGHT, ON_MUG, ON_BOX_LEFT, ON_NOTHING)


def test_a_barcode_belongs_to_the_smallest_box_holding_its_center():
    assert owner(ON_MUG, OBJECTS) == MUG
    assert owner(ON_BOX_LEFT, OBJECTS) == BOX


def test_a_barcode_outside_every_box_has_no_owner():
    assert owner(ON_NOTHING, OBJECTS) is None
    assert owner(ON_MUG, ()) is None


def test_assign_pairs_every_barcode_in_order():
    owners = [holder for _, holder in assign(BARCODES, OBJECTS)]
    assert owners == [BOX, MUG, BOX, None]


def test_a_value_gives_the_object_it_is_printed_on_whatever_it_says():
    assert relate("Shoe", OBJECTS, BARCODES) == Relation("Shoe", ("mug",))
    assert relate("7Q-41", OBJECTS, BARCODES) == Relation("7Q-41", ("box",))


def test_a_name_gives_all_its_values_from_left_to_right():
    assert relate("box", OBJECTS, BARCODES) == Relation("box", ("Kz9", "7Q-41"))


@pytest.mark.parametrize("query", ["box", "BOX", "  Box  "])
def test_names_ignore_case_and_spaces(query):
    assert relate(query, OBJECTS, BARCODES).answers == ("Kz9", "7Q-41")


def test_box_is_a_name_and_shoe_value_a_value():
    objects = (thing("box", 0, 0, 100, 100), thing("shoe", 200, 0, 300, 100))
    barcodes = (code("Box", 50, 50), code("Shoe", 250, 50))
    assert relate("box", objects, barcodes) == Relation("box", ("Box",))
    assert relate("Shoe", objects, barcodes) == Relation("Shoe", ("shoe",))


def test_an_exact_value_wins_over_a_name():
    barcodes = (code("mug", 250, 200),)
    assert relate("mug", OBJECTS, barcodes) == Relation("mug", ("box",))


def test_every_object_with_the_name_contributes_its_barcodes():
    objects = (thing("box", 0, 0, 100, 100), thing("box", 200, 0, 300, 100))
    barcodes = (code("B", 250, 50), code("A", 50, 50))
    assert relate("box", objects, barcodes).answers == ("A", "B")


def test_a_value_read_twice_is_answered_once():
    barcodes = (code("A", 150, 200), code("A", 350, 200))
    assert relate("box", OBJECTS, barcodes).answers == ("A",)
    assert relate("A", OBJECTS, barcodes).answers == ("box",)


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("scissors", NO_BARCODE),
        ("bottle", NO_MATCH),
        ("NOT-A-VALUE", NO_MATCH),
        ("LOOSE", NO_OWNER),
    ],
)
def test_empty_answers_come_with_their_reason_and_no_exception(query, reason):
    assert relate(query, OBJECTS, BARCODES) == Relation(query, (), reason)


def test_nothing_detected_gives_no_match():
    assert relate("box", (), ()) == Relation("box", (), NO_MATCH)


def test_merge_keeps_the_requested_objects_and_drops_their_duplicates():
    requested = (thing("carton", 105, 95, 405, 305),)
    merged = merge_objects(requested, OBJECTS)
    assert merged == (requested[0], TABLE, MUG, SCISSORS)


def test_a_requested_name_outside_the_vocabulary_still_gets_its_codes():
    merged = merge_objects((thing("carton", 105, 95, 405, 305),), OBJECTS)
    assert relate("carton", merged, BARCODES).answers == ("Kz9", "7Q-41")
    assert relate("Shoe", merged, BARCODES).answers == ("mug",)
