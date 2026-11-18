"""Tests for image loading and saving, using small images created on the fly."""

import pytest
from PIL import Image

from shelfvision.images import load_image, save_image

EXIF_ORIENTATION_TAG = 0x0112
ROTATE_90_CLOCKWISE = 6
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def test_alpha_image_becomes_rgb_over_a_light_background(tmp_path):
    path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((0, 0), (*RED, 255))
    image.save(path)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == RED
    assert loaded.getpixel((3, 3)) == WHITE


def test_palette_image_with_transparency_is_flattened(tmp_path):
    path = tmp_path / "palette.png"
    Image.new("P", (2, 2), 0).save(path, transparency=0)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == WHITE


def test_grayscale_image_becomes_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (3, 2), 128).save(path)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    assert loaded.size == (3, 2)
    assert loaded.getpixel((0, 0)) == (128, 128, 128)


def test_path_with_spaces_and_accents(tmp_path):
    path = tmp_path / "carpeta con espacios ñá" / "imagen ó.png"
    image = Image.new("RGB", (5, 7), BLUE)
    assert save_image(image, path) == path
    assert load_image(path).getpixel((0, 0)) == BLUE


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 20), BLUE)
    image.paste(RED, (0, 0, 20, 20))
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = ROTATE_90_CLOCKWISE
    image.save(path, exif=exif, quality=100)
    loaded = load_image(path)
    assert loaded.size == (20, 40)
    red, _, blue = loaded.getpixel((10, 5))
    assert red > blue


def test_loaded_image_does_not_keep_the_file_open(tmp_path):
    path = tmp_path / "closed.png"
    Image.new("RGB", (2, 2)).save(path)
    image = load_image(path)
    path.unlink()
    assert image.getpixel((0, 0)) == (0, 0, 0)


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(tmp_path / "missing.jpg")


def test_folder_is_not_an_image(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(tmp_path)


def test_file_that_is_not_an_image_gives_a_clear_error(tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="Cannot read image"):
        load_image(path)


def test_thin_colored_line_keeps_its_color_in_a_jpeg(tmp_path):
    image = Image.new("RGB", (64, 64), WHITE)
    image.paste(RED, (0, 31, 64, 32))
    red, green, blue = load_image(save_image(image, tmp_path / "line.jpg")).getpixel((32, 31))
    assert red > 200
    assert green < 80
    assert blue < 80
