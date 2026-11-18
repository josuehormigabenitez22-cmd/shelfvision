"""Hand-drawn object boxes and the printed labels of the example photos, used only for benchmarking.

Boxes are (x0, y0, x1, y1) on a 1600-pixel-wide view of each photo; SCALE maps them to the 3840-pixel
originals. They are coarse, which is enough for IoU >= 0.5. A scene may hold several objects with one
name (eight mugs in input_1), so each photo has a list of (name, box) pairs, not one box per name.
The labels (value, owner and corners) come from examples/input/labels.json, written by
tools/make_examples.py when it added them to the photos.
"""

import json
from pathlib import Path

SCALE = 3840 / 1600

LABELS_FILE = Path(__file__).parents[1] / "examples" / "input" / "labels.json"

_VIEW_BOXES = {
    "input_1.jpg": (
        ("mug", (690, 712, 888, 945)),
        ("mug", (895, 638, 1075, 838)),
        ("mug", (1085, 570, 1245, 740)),
        ("mug", (1295, 548, 1485, 708)),
        ("mug", (497, 645, 628, 790)),
        ("mug", (553, 705, 705, 865)),
        ("mug", (627, 582, 750, 705)),
        ("mug", (757, 627, 875, 725)),
        ("glass", (1335, 668, 1467, 908)),
        ("cloth", (620, 395, 1215, 660)),
    ),
    "input_2.jpg": (
        ("phone", (40, 503, 330, 898)),
        ("book", (458, 365, 1205, 935)),
        ("pencil", (1135, 435, 1315, 832)),
        ("glasses", (1300, 578, 1600, 880)),
        ("laptop", (1200, 0, 1600, 715)),
        ("plant", (0, 0, 430, 252)),
    ),
    "input_3.jpg": (
        ("cup", (442, 95, 745, 400)),
        ("laptop", (770, 0, 1470, 418)),
        ("tablet", (965, 362, 1522, 1035)),
        ("glasses", (668, 375, 950, 470)),
        ("book", (0, 320, 385, 920)),
        ("book", (315, 460, 990, 1060)),
    ),
    "input_4.jpg": (
        ("monitor", (30, 0, 1170, 770)),
        ("keyboard", (865, 730, 1395, 1005)),
        ("mouse", (1390, 612, 1585, 682)),
        ("mouse pad", (1222, 575, 1600, 748)),
        ("phone", (498, 820, 628, 920)),
        ("pen", (950, 620, 1158, 685)),
        ("book", (1050, 255, 1430, 455)),
    ),
    "input_5.jpg": (
        ("mug", (170, 200, 475, 505)),
        ("laptop", (668, 0, 1600, 800)),
        ("paper", (150, 610, 1330, 915)),
        ("pen", (822, 655, 1275, 695)),
        ("ruler", (722, 690, 1600, 760)),
        ("cutting board", (0, 450, 620, 625)),
    ),
    "input_6.jpg": (
        ("laptop", (812, 108, 1265, 603)),
        ("lamp", (455, 400, 557, 642)),
        ("plant", (655, 138, 772, 312)),
        ("plant", (1262, 163, 1370, 317)),
        ("plant", (1390, 158, 1548, 317)),
        ("mouse", (325, 762, 438, 845)),
        ("mouse pad", (125, 745, 705, 908)),
        ("book", (795, 710, 1475, 958)),
        ("pen", (1112, 790, 1335, 855)),
    ),
    "input_7.jpg": (
        ("monitor", (648, 100, 1095, 795)),
        ("keyboard", (365, 705, 705, 900)),
        ("mouse", (645, 940, 830, 1022)),
        ("pen", (1105, 688, 1175, 965)),
        ("box", (1290, 828, 1540, 1003)),
    ),
}

GROUND_TRUTH = {
    image: tuple((name, tuple(value * SCALE for value in box)) for name, box in boxes)
    for image, boxes in _VIEW_BOXES.items()
}

# Names that appear in at least one photo, asked for in the query benchmarks
CLASSES = sorted({name for boxes in _VIEW_BOXES.values() for name, _ in boxes})


def _load_labels() -> dict[str, list[dict]]:
    return json.loads(LABELS_FILE.read_text(encoding="utf-8")) if LABELS_FILE.exists() else {}


LABELS = _load_labels()

# Values printed on each photo, and the object each one is stuck on; the solution never uses them
OBJECT_CODES = {image: tuple(label["value"] for label in labels) for image, labels in LABELS.items()}
CODE_OWNERS = {label["value"]: label["owner"] for labels in LABELS.values() for label in labels}
ALL_CODES = frozenset(CODE_OWNERS)


def label_center(image: str, value: str) -> tuple[float, float]:
    """Center of a label in photo pixels, the mean of its four corners"""
    corners = next(label["corners"] for label in LABELS[image] if label["value"] == value)
    return sum(x for x, _ in corners) / 4, sum(y for _, y in corners) / 4


def owner_box(image: str, value: str) -> tuple[float, float, float, float]:
    """The hand-drawn box of the object a label is stuck on: the smallest box of its owner's name
    that holds the label's center, so that a label on one of eight mugs points at that mug"""
    x, y = label_center(image, value)
    boxes = [
        box for name, box in GROUND_TRUTH[image]
        if name == CODE_OWNERS[value] and box[0] <= x <= box[2] and box[1] <= y <= box[3]
    ]
    return min(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
