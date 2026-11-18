"""Image loading and saving through pathlib, so any folder name works on any platform"""

from pathlib import Path
from typing import Final

from PIL import Image, ImageOps

# Printed labels are light, so transparent areas should not turn into dark bars
BACKGROUND_FOR_TRANSPARENCY: Final = (255, 255, 255, 255)
JPEG_QUALITY: Final = 90
# Keeps the full color resolution; with less, thin colored lines fade into the photo
JPEG_SUBSAMPLING: Final = 0


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert to RGB, flattening transparency instead of exposing hidden colours"""
    if not image.has_transparency_data:
        return image.convert("RGB")
    background = Image.new("RGBA", image.size, BACKGROUND_FOR_TRANSPARENCY)
    return Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")


def load_image(path: Path) -> Image.Image:
    """Return the image as upright RGB, fully loaded, so the file is already closed"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as image:
            return _to_rgb(ImageOps.exif_transpose(image))  # phone photos keep their rotation in the EXIF
    except OSError as error:
        raise ValueError(f"Cannot read image {path}: {error}") from error


def save_image(image: Image.Image, path: Path) -> Path:
    """Save the image, creating its folder if needed, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    return path
