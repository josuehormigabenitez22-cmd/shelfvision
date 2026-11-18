"""Draw detections, barcodes and the prompt on an image, with every size proportional to the image."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil, hypot
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from shelfvision.datatypes import ALL_OBJECTS, Barcode, Box, DetectedObject, Point, normalize_query

LINE_WIDTH_FRACTION: Final = 0.003
# A thinner line is too hard to see on a small image
MIN_LINE_WIDTH: Final = 2
FONT_SIZE_FRACTION: Final = 0.022
# Below this size the text cannot be read, however small the image is
MIN_FONT_SIZE: Final = 12
TEXT_COLOR: Final = (255, 255, 255)
BANNER_COLOR: Final = (20, 20, 20)
# Dark enough for white text, and different enough to tell objects apart
PALETTE: Final = (
    (214, 39, 40), (31, 119, 180), (44, 130, 44), (148, 103, 189),
    (190, 90, 20), (200, 40, 140), (20, 130, 140), (90, 90, 90),
)
# Bright yellow stands out on white labels and on the dark parts of the objects alike
ARROW_COLOR: Final = (255, 210, 0)
# The arrow is twice as thick as the outlines, and its head takes a quarter of its length,
# but never less than a few line widths, so that a short arrow still shows its direction
ARROW_WIDTH_LINES: Final = 2
ARROW_HEAD_FRACTION: Final = 0.25
ARROW_HEAD_LINES: Final = 5
# Opacity of an outline's fill, out of 255: the object stays visible under its color
OUTLINE_ALPHA: Final = 100


@dataclass(frozen=True, slots=True)
class Style:
    """The sizes used on one image"""

    line_width: int
    padding: int
    font: ImageFont.FreeTypeFont

    @property
    def text_height(self) -> int:
        return sum(self.font.getmetrics())


def style_for(image_size: tuple[int, int]) -> Style:
    """Scale every size with the short side, so that images of any shape look the same"""
    short_side = min(image_size)
    # Lines and text grow with the image, with a minimum so they can still be seen on small images
    font_size = max(MIN_FONT_SIZE, round(short_side * FONT_SIZE_FRACTION))
    line_width = max(MIN_LINE_WIDTH, round(short_side * LINE_WIDTH_FRACTION))
    return Style(line_width, max(1, font_size // 4), ImageFont.load_default(size=font_size))


def place_label(
    box: Box, label_size: tuple[int, int], image_size: tuple[int, int], taken: Sequence[Box]
) -> Box:
    """Return a place for a label: next to its box if possible, and always inside the image"""
    width, height = label_size
    x0 = max(0, min(box.x0, image_size[0] - width))
    highest_y = max(0, image_size[1] - height)
    preferred = (box.y0 - height, box.y0, box.y1, box.y1 - height)
    sliding = range(max(0, round(box.y0)), highest_y, max(1, height))
    candidates = [Box(x0, y, x0 + width, y + height) for y in (*preferred, *sliding) if 0 <= y <= highest_y]
    free = (label for label in candidates if not any(label.iou(other) > 0 for other in taken))
    fallback = Box(x0, min(max(0, box.y0), highest_y), x0 + width, min(max(0, box.y0), highest_y) + height)
    return next(free, fallback)


def _colors_in_order(names: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """Give each name a color in order of appearance, so that equal inputs give equal images"""
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(dict.fromkeys(names))}


def _draw_label(
    image: Image.Image, near: Box, text: str, color: tuple[int, int, int], taken: list[Box]
) -> None:
    """Draw the text on a band of the color next to the box, away from the labels already taken"""
    draw, style = ImageDraw.Draw(image), style_for(image.size)
    size = (ceil(style.font.getlength(text)) + 2 * style.padding, style.text_height + 2 * style.padding)
    label = place_label(near, size, image.size, taken)
    taken.append(label)
    draw.rectangle((label.x0, label.y0, label.x1, label.y1), fill=color)
    draw.text((label.x0 + style.padding, label.y0 + style.padding), text, TEXT_COLOR, style.font)


def fill_outlines(image: Image.Image, objects: Sequence[DetectedObject], colors: dict[str, tuple[int, int, int]]) -> Image.Image:
    """Return the image with each object's outline filled in its color, half transparent"""
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for item in objects:
        for polygon in item.outline.polygons if item.outline else ():
            draw.polygon(polygon, fill=(*colors[item.name], OUTLINE_ALPHA))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def draw_objects(image: Image.Image, objects: Sequence[DetectedObject]) -> Image.Image:
    """Return a copy of the image with the box, name and score of each object
    Objects with an outline, which exist only with segmentation, also get it filled in
    """
    colors = _colors_in_order(item.name for item in objects)
    has_outlines = any(item.outline for item in objects)
    result = fill_outlines(image, objects, colors) if has_outlines else image.copy()
    draw = ImageDraw.Draw(result)
    style = style_for(image.size)
    for item in objects:
        corners = (item.box.x0, item.box.y0, item.box.x1, item.box.y1)
        draw.rectangle(corners, outline=colors[item.name], width=style.line_width)
    # Labels go after all the boxes, so that no box is drawn over a label
    taken: list[Box] = []
    for item in objects:
        _draw_label(result, item.box, f"{item.name} {item.score:.2f}", colors[item.name], taken)
    return result


def arrow_head(start: Point, tip: Point, size: float) -> tuple[Point, Point, Point] | None:
    """Return the triangle of an arrow head of the given length at the tip, None for a zero arrow."""
    (x0, y0), (x1, y1) = start, tip
    length = hypot(x1 - x0, y1 - y0)
    if length == 0:
        return None
    along_x, along_y = (x1 - x0) / length, (y1 - y0) / length
    back_x, back_y = x1 - along_x * size, y1 - along_y * size
    side_x, side_y = -along_y * size / 2, along_x * size / 2
    return tip, (back_x + side_x, back_y + side_y), (back_x - side_x, back_y - side_y)


def _draw_arrow(draw: ImageDraw.ImageDraw, style: Style, start: Point, tip: Point | None) -> None:
    """Draw a dot where the arrow starts, so that one pointing at the camera still shows, and the arrow"""
    width = ARROW_WIDTH_LINES * style.line_width
    x, y = start
    draw.ellipse((x - width, y - width, x + width, y + width), fill=ARROW_COLOR)
    if tip is None:
        return
    length = hypot(tip[0] - x, tip[1] - y)
    head_size = min(length, max(ARROW_HEAD_LINES * style.line_width, ARROW_HEAD_FRACTION * length))
    if head := arrow_head(start, tip, head_size):
        draw.line((start, tip), fill=ARROW_COLOR, width=width)
        draw.polygon(head, fill=ARROW_COLOR)


def draw_barcodes(
    image: Image.Image, barcodes: Sequence[Barcode], tips: Sequence[Point | None]
) -> Image.Image:
    """Return a copy of the image with each barcode's outline, value and normal arrow
    tips holds, for each barcode in order, where its arrow ends, or None when it has no normal.
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)
    style = style_for(image.size)
    colors = _colors_in_order(barcode.value for barcode in barcodes)
    for barcode in barcodes:
        draw.polygon(barcode.corners, outline=colors[barcode.value], width=style.line_width)
    taken: list[Box] = []
    for barcode in barcodes:
        _draw_label(result, barcode.box, barcode.value, colors[barcode.value], taken)
    # Arrows go last, so that no label hides them
    for barcode, tip in zip(barcodes, tips, strict=True):
        _draw_arrow(draw, style, barcode.center, tip)
    return result


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Split the text into lines no wider than max_width, unless a single word is wider"""
    lines = [""]
    for word in text.split():
        joined = f"{lines[-1]} {word}".strip()
        if font.getlength(joined) <= max_width or not lines[-1]:
            lines[-1] = joined
        else:
            lines.append(word)
    return lines


def add_banner(image: Image.Image, text: str) -> Image.Image:
    """Return the image with the text in a band above it, so that the band hides nothing"""
    style = style_for(image.size)
    lines = wrap_text(text, style.font, image.width - 2 * style.padding)
    height = len(lines) * style.text_height + 2 * style.padding
    result = Image.new("RGB", (image.width, image.height + height), BANNER_COLOR)
    result.paste(image, (0, height))
    draw = ImageDraw.Draw(result)
    for number, line in enumerate(lines):
        position = (style.padding, style.padding + number * style.text_height)
        draw.text(position, line, TEXT_COLOR, style.font)
    return result


def describe_query(query: str | Iterable[str]) -> str:
    """Return the query as the user should read it"""
    names = normalize_query(query)
    return ALL_OBJECTS if names == ALL_OBJECTS else ", ".join(names) or "(empty)"


def render_detections(image: Image.Image, objects: Sequence[DetectedObject], query: str | Iterable[str]) -> Image.Image:
    """Return the detection image: the boxes on the image, with the prompt above it"""
    summary = f"Prompt: {describe_query(query)} | Found: {len(objects)}"  # text shown in the banner
    return add_banner(draw_objects(image, objects), summary)


def render_barcodes(
    image: Image.Image, barcodes: Sequence[Barcode], tips: Sequence[Point | None], query: str | Iterable[str]
) -> Image.Image:
    """Return the barcode image: outlines, values and normal arrows, with the prompt above"""
    summary = f"Prompt: {describe_query(query)} | Barcodes: {len(barcodes)}"
    return add_banner(draw_barcodes(image, barcodes, tips), summary)
