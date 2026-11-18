"""Build the example photos: public-domain desk photos with Code128 labels added in perspective.

The photos come from Wikimedia Commons, all under CC0 (see examples/input/SOURCES.md). None of them
shows a barcode, so this script prints labels on flat parts of the objects: each label is drawn sharp,
warped onto four corners picked by hand on the photo, shaded with the light of the surface under it,
slightly blurred and given sensor noise, and the photo is saved as JPEG. The values, corners and the
object each label is stuck on are written to examples/input/labels.json, the ground truth of the
benchmarks.

Run it from the project root with: pixi run python tools/make_examples.py --raw <folder with the photos>
The folder holds the photos as downloaded, named scene_<n>.jpg; see SOURCES for their pages.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from shelfvision.code128 import CodeSet, encode, symbols_to_widths

OUTPUT: Final = Path(__file__).parents[1] / "examples" / "input"
# Corners were picked on a 1600-pixel-wide view of each 3840-pixel photo
VIEW_SCALE: Final = 3840 / 1600
MODULE_PX: Final = 8  # label drawn at 8 pixels per module, then warped down
QUIET_MODULES: Final = 10
BAR_HEIGHT_FRACTION: Final = 0.78
PAPER: Final = 238.0
INK: Final = 28.0
BLUR_SIGMA: Final = 0.7
NOISE_SIGMA: Final = 2.5
JPEG_QUALITY: Final = 92
LIGHT_WINDOW: Final = 301  # pixels around a label that tell how much light falls on it
SEED: Final = 2026

COMMONS: Final = "https://commons.wikimedia.org/wiki/File:"
# Photo, its Wikimedia Commons file and author; all CC0, downloaded at 3840 pixels wide
SOURCES: Final = {
    "scene_1.jpg": ("Enamel_cups_on_a_table_(Unsplash).jpg", "André Freitas"),
    "scene_2.jpg": ("White_work_table_with_notes,_smartphone_and_laptop_(Unsplash).jpg", "JESHOOTS.COM"),
    "scene_4.jpg": ("Journaling_over_coffee_(Unsplash).jpg", "Thought Catalog"),
    "scene_5.jpg": ("Desk_Setup_(Unsplash).jpg", "Luke Chesser"),
    "scene_6.jpg": ("Macbook_and_tea_(Unsplash).jpg", "Olu Eletu"),
    "scene_7.jpg": ("Notepad_on_a_busy_workspace_(Unsplash).jpg", "Grovemade"),
    "scene_8.jpg": ("Winter_Workspace_(Unsplash).jpg", "Norbert Levajsics"),
}


@dataclass(frozen=True, slots=True)
class Label:
    """A label: its value, the object it is stuck on, its center, axes and size in view pixels

    along is the direction of the scan line (across the bars), up the direction of the bars; they
    need not be perpendicular, so a label on a surface seen at an angle becomes a parallelogram.
    """

    value: str
    owner: str
    center: tuple[float, float]
    along: tuple[float, float]
    up: tuple[float, float]
    length: float
    height: float

    def corners(self) -> np.ndarray:
        """Top-left, top-right, bottom-right and bottom-left corners in photo pixels"""
        center = np.array(self.center) * VIEW_SCALE
        along = np.array(self.along) / np.linalg.norm(self.along) * self.length * VIEW_SCALE / 2
        up = np.array(self.up) / np.linalg.norm(self.up) * self.height * VIEW_SCALE / 2
        return np.array([center - along - up, center + along - up, center + along + up, center - along + up])


# (photo, output name) and its labels; owners use the names of benchmarks/ground_truth.py
SCENES: Final = (
    ("scene_1.jpg", "input_1.jpg", (
        Label("MG214", "mug", (1010, 738), (0, 1), (1, 0), 165, 52),
        Label("MG380", "mug", (733, 822), (0, 1), (1, 0), 158, 50),
        Label("MG57", "mug", (1372, 628), (0, 1), (1, 0), 128, 44),
        Label("CL903", "cloth", (905, 515), (1, -0.03), (-0.25, -1), 200, 58),
    )),
    ("scene_2.jpg", "input_2.jpg", (
        Label("PH118", "phone", (186, 700), (-100, 370), (370, 100), 225, 62),
        Label("NB775", "book", (1012, 600), (1, -0.03), (0.03, 1), 210, 64),
        Label("X001NB42K7", "book", (650, 805), (1, -0.05), (0.05, 1), 200, 60),
        Label("LT63", "laptop", (1335, 395), (295, 290), (-290, 295), 140, 44),
    )),
    ("scene_4.jpg", "input_3.jpg", (
        Label("LT509", "laptop", (935, 195), (1, 0), (0, 1), 160, 52),
        Label("TB246", "tablet", (1240, 720), (355, 135), (-135, 355), 220, 70),
        Label("BK831", "book", (215, 560), (0.357, 0.934), (-0.934, 0.357), 190, 60),
        Label("NB320", "book", (835, 735), (0.866, 0.5), (-0.5, 0.866), 160, 52),
    )),
    ("scene_5.jpg", "input_4.jpg", (
        Label("MN725", "monitor", (450, 600), (0.923, -0.385), (0.324, 0.946), 200, 62),
        Label("PD19", "mouse pad", (1340, 705), (0.993, 0.118), (0.864, -0.503), 160, 46),
    )),
    ("scene_6.jpg", "input_5.jpg", (
        Label("MG466", "mug", (255, 370), (0, 1), (1, 0), 190, 56),
        Label("LT902", "laptop", (1290, 262), (1, 0.03), (-0.05, 1), 180, 56),
        Label("PP137", "paper", (500, 790), (1, 0.08), (0.33, -0.94), 210, 46),
    )),
    ("scene_7.jpg", "input_6.jpg", (
        Label("LT388", "laptop", (1040, 320), (1, 0), (0, 1), 200, 60),
        Label("PL71", "plant", (1468, 255), (1, 0), (0, 1), 118, 40),
        Label("PD650", "mouse pad", (470, 865), (1, -0.01), (0.46, -0.89), 170, 44),
        Label("NB204", "book", (1290, 872), (1, -0.1), (0.46, -0.89), 150, 40),
    )),
    ("scene_8.jpg", "input_7.jpg", (
        Label("MN513", "monitor", (800, 435), (0.906, 0.423), (-0.092, 0.996), 185, 58),
        Label("BX847", "box", (1435, 942), (0.995, 0.102), (0, 1), 160, 40),
    )),
)


def draw_label(value: str) -> np.ndarray:
    """Return a sharp gray label: paper, quiet zones and bars, at MODULE_PX pixels per module"""
    widths = symbols_to_widths(encode([(CodeSet.B, value)]))
    modules = sum(widths) + 2 * QUIET_MODULES
    width = modules * MODULE_PX
    height = round(width / 3.2)
    label = np.full((height, width), PAPER, np.float32)
    bar_top = round(height * (1 - BAR_HEIGHT_FRACTION) / 2)
    x = QUIET_MODULES * MODULE_PX
    for index, modules_wide in enumerate(widths):
        if index % 2 == 0:  # bars at even positions, spaces at odd ones
            label[bar_top : height - bar_top, x : x + modules_wide * MODULE_PX] = INK
        x += modules_wide * MODULE_PX
    return label


def paste(photo: np.ndarray, label: Label, rng: np.random.Generator) -> np.ndarray:
    """Warp the label onto its corners and blend it with the light of the surface under it"""
    sharp = draw_label(label.value)
    height, width = sharp.shape
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = label.corners().astype(np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    size = (photo.shape[1], photo.shape[0])
    gray = cv2.warpPerspective(sharp, transform, size, flags=cv2.INTER_AREA)
    coverage = cv2.warpPerspective(np.ones_like(sharp), transform, size, flags=cv2.INTER_AREA)
    # White paper shows the light that falls on it, not the color of what it is stuck on: a label on a
    # black screen is still white. The light is the envelope of the brightest pixels around the label
    light_map = cv2.dilate(photo, np.ones((LIGHT_WINDOW, LIGHT_WINDOW), np.uint8))
    light_map = cv2.GaussianBlur(light_map, (0, 0), LIGHT_WINDOW / 3)
    light = np.clip(light_map.mean(axis=2, keepdims=True) / 235.0, 0.55, 1.05)
    tint = light_map / np.maximum(light_map.mean(axis=2, keepdims=True), 1.0)
    tint = 0.8 + 0.2 * tint  # keep most of the paper white
    printed = gray[..., None] * light * tint
    printed = cv2.GaussianBlur(printed, (0, 0), BLUR_SIGMA)
    printed += rng.normal(0.0, NOISE_SIGMA, printed.shape)
    alpha = cv2.GaussianBlur(coverage, (0, 0), 0.6)[..., None]
    return photo * (1 - alpha) + printed * alpha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="folder with the downloaded photos")
    folder = parser.parse_args().raw
    rng = np.random.default_rng(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    for source, name, labels in SCENES:
        photo = cv2.imread(str(folder / source), cv2.IMREAD_COLOR).astype(np.float32)
        for label in labels:
            photo = paste(photo, label, rng)
        cv2.imwrite(str(OUTPUT / name), np.clip(photo, 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        truth[name] = [
            {"value": label.value, "owner": label.owner, "corners": np.round(label.corners(), 1).tolist()}
            for label in labels
        ]
        print(f"{name}: {len(labels)} labels")
    (OUTPUT / "labels.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    rows = [
        f"| `{name}` | [{SOURCES[source][0]}]({COMMONS}{SOURCES[source][0]}) | {SOURCES[source][1]} | CC0 |"
        for source, name, _ in SCENES
    ]
    (OUTPUT / "SOURCES.md").write_text(
        "# Sources of the example photos\n\n"
        "Every photo is in the public domain (CC0) on Wikimedia Commons. The Code128 labels were added\n"
        "by `tools/make_examples.py`; the photos are otherwise unchanged.\n\n"
        "| Photo | Wikimedia Commons file | Author | License |\n|---|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
