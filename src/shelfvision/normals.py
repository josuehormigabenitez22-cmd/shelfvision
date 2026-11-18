"""Surface normals of the barcodes, from MoGe-2's normal map of the whole image.

Coordinates follow MoGe and OpenCV: x to the right, y down and z away from the camera, so a label
that faces the camera has a normal with a negative z. The intrinsics come from MoGe too, because
the photos carry none in their EXIF.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np
import torch
from PIL import Image

from shelfvision.datatypes import Barcode, Point, Vector3
from shelfvision.devices import AUTO, resolve_device

if TYPE_CHECKING:
    from moge.model.v2 import MoGeModel

MODEL_ID: Final = "Ruicheng/moge-2-vits-normal"
MODEL_REVISION: Final = "26b477f41595707c5db6770294c0d1721e8ed4ed"
# Width of the image MoGe sees and its lowest resolution level: about 7 s per image on a CPU
INPUT_WIDTH: Final = 1280
RESOLUTION_LEVEL: Final = 0
# The arrow is this many label widths long in 3D, so it grows and shrinks with the label
ARROW_LABEL_WIDTHS: Final = 1.5
# MoGe asks for float32 autocast, which a CPU cannot do; torch warns and runs without it
AUTOCAST_WARNING: Final = "In CPU autocast, but the target dtype is not supported"


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Pinhole camera, in pixels of the original image"""
    fx: float
    fy: float
    cx: float
    cy: float

    def ray(self, point: Point) -> np.ndarray:
        """Direction from the camera through an image point, scaled so that its z is 1"""
        x, y = point
        return np.array([(x - self.cx) / self.fx, (y - self.cy) / self.fy, 1.0])

    def project(self, xyz: np.ndarray) -> Point:
        x, y, z = xyz.tolist()
        return self.fx * x / z + self.cx, self.fy * y / z + self.cy


@dataclass(frozen=True, slots=True)
class SurfaceMap:
    """MoGes unit normals for one image, as rows x cols x 3, smaller than the image by scale"""

    normals: np.ndarray
    scale: float
    intrinsics: Intrinsics


def region_normal(normals: np.ndarray, polygon: Sequence[Point], scale: float) -> np.ndarray | None:
    """Return the unit median normal inside a polygon in original pixels, None if it covers no pixel"""
    # Paint the label on a mask the size of the normal map
    mask = np.zeros(normals.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(polygon) * scale).astype(np.int32)], 1)
    samples = normals[mask > 0]
    if not len(samples):
        return None
    # The median per axis ignores the pixels that spill over the label's edges
    median = np.median(samples, axis=0)
    length = float(np.linalg.norm(median))
    return median / length if length > 0 else None


def facing_camera(normal: np.ndarray, ray: np.ndarray) -> np.ndarray:
    """Flip a normal that points away from the camera: a visible label always faces it"""
    return -normal if float(normal @ ray) > 0 else normal


def _normal_of(barcode: Barcode, surface: SurfaceMap) -> Vector3 | None:
    normal = region_normal(surface.normals, barcode.corners, surface.scale)
    if normal is None:
        return None
    x, y, z = facing_camera(normal, surface.intrinsics.ray(barcode.center)).tolist()
    return x, y, z


def with_normals(barcodes: Sequence[Barcode], surface: SurfaceMap) -> tuple[Barcode, ...]:
    """Return the barcodes with their unit normal, facing the camera; None where the map has no pixel"""
    return tuple(replace(barcode, normal=_normal_of(barcode, surface)) for barcode in barcodes)


def label_width(barcode: Barcode) -> float:
    """Length in pixels of the longest side of the labels outline"""
    corners = np.asarray(barcode.corners)
    return float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).max())


def arrow_tip(barcode: Barcode, intrinsics: Intrinsics) -> Point | None:
    """Return where the 3D arrow along the label's normal ends in the image, None without a normal.

    The arrow starts at the label's center and is ARROW_LABEL_WIDTHS label widths long. Its
    projection does not depend on the label's depth, which a single image cannot give: placing the
    label at depth 1 scales the label and the arrow alike.
    """
    if barcode.normal is None:
        return None
    length = ARROW_LABEL_WIDTHS * label_width(barcode) / intrinsics.fx
    tip = intrinsics.ray(barcode.center) + length * np.asarray(barcode.normal)
    # A tip behind the camera cannot be drawn; it would need a label wider than its distance
    return intrinsics.project(tip) if tip[2] > 0 else None


class NormalEstimator:
    """MoGe 2, loaded on first use. moge is imported only then, so the rest works without it"""

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)

    @cached_property
    def _model(self) -> "MoGeModel":
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as error:
            raise ImportError(
                "Surface normals need the moge package, which installs on Windows and Linux only"
            ) from error
        return MoGeModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(self.device).eval()

    def estimate(self, image: Image.Image) -> SurfaceMap:
        """Return MoGe's normal map of an RGB image and the camera it implies."""
        width, height = image.size
        # MoGe works at 1280 px wide; the aspect ratio is kept
        pixels = cv2.resize(np.asarray(image), (INPUT_WIDTH, INPUT_WIDTH * height // width),
                            interpolation=cv2.INTER_AREA)
        tensor = (torch.from_numpy(pixels).permute(2, 0, 1).float() / 255).to(self.device)
        # use_fp16 stays off on a GPU too, so that both devices give the same normals
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=AUTOCAST_WARNING, category=UserWarning)
            output = self._model.infer(tensor, resolution_level=RESOLUTION_LEVEL, use_fp16=False)
        normals = output["normal"].cpu().numpy()  # H x W x 3, a unit normal per pixel
        # MoGe gives the intrinsics as fractions of the image width and height
        (fx, _, cx), (_, fy, cy), _ = output["intrinsics"].cpu().tolist()
        intrinsics = Intrinsics(fx * width, fy * height, cx * width, cy * height)
        return SurfaceMap(normals, normals.shape[1] / width, intrinsics)
