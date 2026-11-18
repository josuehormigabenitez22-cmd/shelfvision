"""Checks that the environment and the example images are ready."""
import json
from pathlib import Path

from moge.model.v2 import MoGeModel
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection

import shelfvision

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "input"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def test_package_is_loaded_from_src():
    assert Path(shelfvision.__file__).parents[1].name == "src"


def test_model_classes_are_available():
    assert MoGeModel and AutoModelForZeroShotObjectDetection


def test_example_images_are_in_place():
    paths = sorted(EXAMPLES.glob("*.jpg"))
    labelled = json.loads((EXAMPLES / "labels.json").read_text(encoding="utf-8"))
    assert [path.name for path in paths] == sorted(labelled)
    assert {image_size(path)[0] for path in paths} == {3840}