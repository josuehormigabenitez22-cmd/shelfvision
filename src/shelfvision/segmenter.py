"""Optional segmentation: the silhouette of each detected object, from SAM 2.1 prompted with its box.

It runs only when the orchestrator is created with segment=True; otherwise this model is never
loaded nor downloaded, and every result is the same as without this module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from transformers.utils import logging as transformers_logging

from shelfvision.datatypes import Box, Outline, Polygon
from shelfvision.devices import AUTO, resolve_device

MODEL_ID: Final = "facebook/sam2.1-hiera-tiny"
MODEL_REVISION: Final = "de431c4043854a71d8101e17995dfe596bf101a5"
# Parts of a mask smaller than this fraction of its largest part are specks, not the object
MIN_PART_FRACTION: Final = 0.05
# Outlines are simplified to this fraction of the image's longer side: about 4 px on a 4K photo
OUTLINE_TOLERANCE: Final = 0.001


@dataclass(frozen=True, slots=True, eq=False)
class ImageEmbedding:
    """What SAM computes once per image, reusable for any set of boxes."""

    features: list[torch.Tensor]
    image_size: tuple[int, int]


def mask_outline(mask: np.ndarray, tolerance: float) -> Outline | None:
    """Return the outer polygons of a binary mask, without specks; None if the mask is empty."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    if not areas or max(areas) == 0:
        return None
    kept = [c for c, area in zip(contours, areas, strict=True) if area >= MIN_PART_FRACTION * max(areas)]
    simplified = (cv2.approxPolyDP(contour, tolerance, closed=True)[:, 0] for contour in kept)
    polygons = tuple(_to_polygon(points) for points in simplified if len(points) >= 3)
    return Outline(polygons) if polygons else None


def _to_polygon(points: np.ndarray) -> Polygon:
    return tuple((float(x), float(y)) for x, y in points)


class Segmenter:
    """SAM 2.1 tiny, loaded on first use, on the chosen device."""

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)

    @cached_property
    def _processor(self) -> Sam2Processor:
        return Sam2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    @cached_property
    def _model(self) -> Sam2Model:
        # The checkpoint is saved as SAM 2's video model, whose image part is exactly Sam2Model;
        # transformers warns about that on every load, so its warnings are hidden while loading
        verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            model = Sam2Model.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        finally:
            transformers_logging.set_verbosity(verbosity)
        return model.to(self.device).eval()

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> ImageEmbedding:
        pixels = self._processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        return ImageEmbedding(self._model.get_image_embeddings(pixels), image.size)

    @torch.inference_mode()
    def outlines(self, embedding: ImageEmbedding, boxes: Sequence[Box]) -> tuple[Outline | None, ...]:
        """Return the outline of the object inside each box, None where SAM finds no object."""
        if not boxes:
            return ()
        width, height = embedding.image_size
        # SAM works on a 1024 px image, so the boxes are rescaled to it
        prompts = self._processor(
            input_boxes=[[[box.x0, box.y0, box.x1, box.y1] for box in boxes]],
            original_sizes=[[height, width]],
            return_tensors="pt",
        )
        output = self._model(
            input_boxes=prompts["input_boxes"].to(self.device),
            image_embeddings=embedding.features,
            multimask_output=False,
        )
        # Back to full-size masks of True and False
        masks = self._processor.post_process_masks(output.pred_masks.cpu(), prompts["original_sizes"])[0]
        tolerance = OUTLINE_TOLERANCE * max(width, height)
        return tuple(mask_outline(mask.numpy(), tolerance) for mask in masks[:, 0])
