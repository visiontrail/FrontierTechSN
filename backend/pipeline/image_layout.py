"""Derive photo layout from the downloaded pixels, including EXIF orientation."""

from pathlib import Path

from PIL import Image


def image_layout(path: Path, *, fit: str = "cover", kind: str = "") -> dict:
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in {5, 6, 7, 8}:
            width, height = height, width
    return {
        "width": width,
        "height": height,
        "fit": "contain" if height > width or kind == "logo" or fit == "contain" else "cover",
    }
