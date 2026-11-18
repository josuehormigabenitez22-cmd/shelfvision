"""Zero-shot object detection with OWLv2, split so that each stage can be cached on its own"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Final

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from shelfvision.datatypes import ALL_OBJECTS, Box, DetectedObject, normalize_query
from shelfvision.devices import AUTO, resolve_device

# Defines the model identifier
MODEL_ID: Final = "google/owlv2-base-patch16-ensemble"
# Pins the model version to ensure reproducible results
MODEL_REVISION: Final = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
# Defines the model input image size
INPUT_SIDE: Final = 960
# Defines the padding color for images
PADDING_COLOR: Final = (127, 127, 127)
# Defines the maximum number of text token
MAX_TOKENS: Final = 16
# Defines the maximum number of cached text results
TEXT_CACHE_SIZE: Final = 256

# These thresholds were first tuned on 8 photos of one scene, then checked on 7 desk photos
MIN_OBJECTNESS: Final = 0.25
# Larger boxes are usually the table or the background
MAX_BOX_FRACTION: Final = 0.35
NMS_IOU: Final = 0.4
MIN_NAME_SCORE: Final = 0.25
UNKNOWN_NAME: Final = "object"
# 0.22 was best on the first scene alone (F1 0.85) but let through many false boxes on the desks
# (F1 0.58); over both sets 0.34 gives the best F1, 0.71 (0.81 and 0.64)
MIN_QUERY_SCORE: Final = 0.34
MIN_RIVAL_RATIO: Final = 1.0
# Avoids dividing by zero when no vocabulary word fits a box
SMALLEST_RIVAL_SCORE: Final = 1e-6
# Words this close to the query are synonyms of it, not competitors
MAX_RIVAL_SIMILARITY: Final = 0.85

NAMING_TEMPLATES: Final = ("a {} on a table",)
RIVAL_TEMPLATES: Final = ("a photo of a {}",)
QUERY_TEMPLATES: Final = ("a photo of a {}", "a {} on a table", "a close-up photo of a {}")
# Common objects, used to name boxes and to reject a query when another word fits better
VOCABULARY: Final = (
    "person", "chair", "table", "laptop", "keyboard", "phone", "book", "pen", "pencil", "paper", "plate",
    "bowl", "cup", "glass", "mug", "bottle", "can", "jar", "box", "bag", "backpack", "hat", "shirt", "shoe",
    "glove", "clock", "lamp", "plant", "flower", "apple", "banana", "orange", "sandwich", "spoon", "fork",
    "knife", "scissors", "hammer", "wrench", "screwdriver", "pliers", "saw", "drill", "tape", "ruler", "toy",
    "ball", "remote", "camera", "headphones", "umbrella", "towel", "cloth",
)


@dataclass(frozen=True, slots=True, eq=False)
class ImageFeatures:
    """What owlv2 computes once per image, reusable for any text query"""
    tokens: torch.Tensor
    boxes: torch.Tensor
    objectness: torch.Tensor
    image_size: tuple[int, int]


def to_pixel_boxes(normalized: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Converts normalized bounding boxes to pixel coordinates in the original image"""
    width, height = image_size
    # Splits the normalized boxes into center coordinates and dimensions
    # cx: horizontal center, cy: vertical center, box_width: width, box_height: height
    cx, cy, box_width, box_height = normalized.unbind(-1)
    half_width, half_height = box_width / 2, box_height / 2
    # Convert to points (x0, y0, x1, y1)
    corners = torch.stack([cx - half_width, cy - half_height, cx + half_width, cy + half_height], -1)
    limits = torch.tensor([width, height, width, height], dtype=corners.dtype)
    return torch.minimum((corners * max(image_size)).clamp(min=0), limits) # pixels convert


def box_areas(boxes: torch.Tensor) -> torch.Tensor:
    """Return the area of each box"""
    return (boxes[:, 2:] - boxes[:, :2]).prod(-1)


def select_objects(
    boxes: torch.Tensor, objectness: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    """Return the indices of the boxes that look like separate objects of any kind"""
    candidates = (objectness >= MIN_OBJECTNESS).nonzero()[:, 0]
    survivors = candidates[nms(boxes[candidates], objectness[candidates], NMS_IOU)]
    areas = box_areas(boxes[survivors])
    largest_area = MAX_BOX_FRACTION * image_size[0] * image_size[1]
    return survivors[(areas > 0) & (areas <= largest_area)]


def select_matches(
    boxes: torch.Tensor, scores: torch.Tensor, rival_ratios: torch.Tensor, min_score: float
) -> torch.Tensor:
    """Return the indices of the boxes that fit the query, and fit it better than any rival"""
    candidates = ((scores >= min_score) & (box_areas(boxes) > 0)).nonzero()[:, 0]
    survivors = candidates[nms(boxes[candidates], scores[candidates], NMS_IOU)]
    return survivors[rival_ratios[survivors] >= MIN_RIVAL_RATIO]


def find_rivals(name: str, similarities: torch.Tensor) -> torch.Tensor:
    """Mark the vocabulary words that compete with a query, leaving out its own words and synonyms"""
    words = name.split()
    not_in_query = torch.tensor([word not in words for word in VOCABULARY])
    return not_in_query & (similarities < MAX_RIVAL_SIMILARITY)


def display_name(word: str, score: float) -> str:
    """Return the vocabulary word, or a neutral name if the model is not confident enough"""
    return word if score >= MIN_NAME_SCORE else UNKNOWN_NAME


def _to_box(values: torch.Tensor) -> Box:
    return Box(*values.tolist())


class ObjectDetector:
    """Defines an object detector that separates image encoding, text encoding and scoring
    The model runs on the chosen device, and everything it returns comes back to the CPU at once,
    so that the selection of boxes below runs the same way on any device.
    """

    def __init__(self, device: str = AUTO) -> None:
        self.device = resolve_device(device)
        # Creates a cache to reuse text embeddings
        self.encode_names = lru_cache(maxsize=TEXT_CACHE_SIZE)(self._embed_names)

    @cached_property
    def _processor(self) -> Owlv2Processor:
        return Owlv2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    @cached_property
    def _model(self) -> Owlv2ForObjectDetection:
        # Loads and caches the pretrained OWLv2 model in evaluation mode
        return Owlv2ForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(self.device).eval()

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses an image into a normalized tensor for the model"""
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), PADDING_COLOR)
        canvas.paste(image, (0, 0))
        resized = canvas.resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255
        settings = self._processor.image_processor
        mean = torch.tensor(settings.image_mean)[:, None, None]
        std = torch.tensor(settings.image_std)[:, None, None]
        return ((pixels - mean) / std)[None].to(self.device)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> ImageFeatures:
        """Extracts reusable image features, bounding boxes and objectness scores"""
        feature_map, _ = self._model.image_embedder(self._to_tensor(image)) # extrac visual characteristic excluding second result _
        batch, rows, columns, depth = feature_map.shape
        tokens = feature_map.reshape(batch, rows * columns, depth)
        # The tokens stay on the device for scoring; the boxes and objectness come back to the CPU
        normalized_boxes = self._model.box_predictor(tokens, feature_map)[0].cpu()
        objectness = torch.sigmoid(self._model.objectness_predictor(tokens))[0].cpu()
        return ImageFeatures(tokens, to_pixel_boxes(normalized_boxes, image.size), objectness, image.size)

    @torch.inference_mode()
    def _embed_names(self, names: tuple[str, ...], templates: tuple[str, ...]) -> torch.Tensor:
        """Return one unit vector per name, averaged over its templates"""
        prompts = [template.format(name) for name in names for template in templates]
        tokens = self._processor.tokenizer(prompts, padding="max_length", max_length=MAX_TOKENS, truncation=True, return_tensors="pt")
        output = self._model.owlv2.get_text_features(input_ids=tokens["input_ids"].to(self.device), attention_mask=tokens["attention_mask"].to(self.device))
        embeddings = output.pooler_output / output.pooler_output.norm(dim=-1, keepdim=True)
        means = embeddings.reshape(len(names), len(templates), -1).mean(1)
        return (means / means.norm(dim=-1, keepdim=True)).cpu()

    @torch.inference_mode()
    def score(self, features: ImageFeatures, embeddings: torch.Tensor) -> torch.Tensor:
        """Return one probability per box and name, with shape (boxes, names)."""
        query_mask = torch.ones(embeddings.shape[0], dtype=torch.bool, device=self.device)[None] # Creates a mask for the active text queries
        logits, _ = self._model.class_predictor(features.tokens, embeddings[None].to(self.device), query_mask) # Computes object classification scores by comparing image and text features
        return torch.sigmoid(logits)[0].cpu() # returns a probability score for each box and text query

    def detect(self, features: ImageFeatures, query: str | Iterable[str]) -> tuple[DetectedObject, ...]:
        """Find the requested objects, or every object if the query is 'all' """
        names = normalize_query(query)
        return self.detect_all(features) if names == ALL_OBJECTS else self.detect_names(features, names)

    def detect_all(self, features: ImageFeatures) -> tuple[DetectedObject, ...]:
        """Find objects without any query, then name each box with the vocabulary"""
        chosen = select_objects(features.boxes, features.objectness, features.image_size)
        word_scores = self.score(features, self.encode_names(VOCABULARY, NAMING_TEMPLATES))
        best_scores, best_words = word_scores[chosen].max(dim=1) # Assigns the best vocabulary label to each selected box
        rows = zip(chosen.tolist(), best_words.tolist(), best_scores.tolist(), strict=True)
        return tuple(
            DetectedObject(display_name(VOCABULARY[word], score), _to_box(features.boxes[index]), score)
            for index, word, score in rows
        ) # Creates the final detected objects with their labels, boxes and scores

    def detect_names(self, features: ImageFeatures, names: tuple[str, ...], min_score: float = MIN_QUERY_SCORE) -> tuple[DetectedObject, ...]:
        """Find the boxes of each name. An absent object should give no boxes for that name"""
        if not names:
            return ()
        vocabulary = self.encode_names(VOCABULARY, RIVAL_TEMPLATES) #encodes the vocabulary for object comparison
        similarities = self.encode_names(names, RIVAL_TEMPLATES) @ vocabulary.T # computes text similarities to identify rival object classes
        word_scores = self.score(features, vocabulary)
        name_scores = self.score(features, self.encode_names(names, QUERY_TEMPLATES))
        found = []
        for column, name in enumerate(names):
            rival_scores = word_scores[:, find_rivals(name, similarities[column])].max(dim=1).values
            ratios = name_scores[:, column] / rival_scores.clamp(min= SMALLEST_RIVAL_SCORE)
            scores = torch.sqrt(name_scores[:, column] * features.objectness)
            chosen = select_matches(features.boxes, scores, ratios, min_score).tolist()
            found += [DetectedObject(name, _to_box(features.boxes[i]), scores[i].item()) for i in chosen]
        return tuple(found)
