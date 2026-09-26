"""Unit tests for MPO image decoding tolerance and aspect-preserving downscaling (#124509)."""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tools.vision_tools import _resize_image_for_vision
from tools.vision_tools_image_prep import _validate_raster_image_decodable


def test_validate_raster_image_decodable_valid_jpeg(tmp_path: Path):
    """Standard valid JPEG validates cleanly."""
    img_path = tmp_path / "valid.jpg"
    img = Image.new("RGB", (200, 200), color="blue")
    img.save(img_path, format="JPEG")

    assert _validate_raster_image_decodable(img_path) is None


def test_validate_raster_mpo_with_stripped_secondary_frame(tmp_path: Path):
    """MPO file whose secondary frame fails to seek/load should not reject the valid primary frame."""
    img_path = tmp_path / "picasa_photo.jpg"
    img = Image.new("RGB", (300, 200), color="green")
    img.save(img_path, format="JPEG")

    # Simulate Pillow opening an MPO file where frame 1 is decoded, but seeking frame 2 raises SyntaxError/ValueError
    mock_frame = MagicMock()
    mock_frame.width = 300
    mock_frame.height = 200
    mock_frame.load.return_value = None

    class MockIterator:
        def __init__(self, im):
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.count == 0:
                self.count += 1
                return mock_frame
            raise SyntaxError("not a JPEG file")

    real_open = Image.open

    def fake_open(p, *args, **kwargs):
        im = real_open(p, *args, **kwargs)
        im.format = "MPO"
        return im

    with patch("PIL.Image.open", side_effect=fake_open), \
         patch("PIL.ImageSequence.Iterator", side_effect=MockIterator):
        err = _validate_raster_image_decodable(img_path)
        assert err is None, f"Expected MPO secondary frame failure to be tolerated, got error: {err}"


def test_validate_raster_non_mpo_corrupt_frame_rejected(tmp_path: Path):
    """For non-MPO images (e.g. GIF or PNG animation), a corrupt frame should still be rejected."""
    img_path = tmp_path / "corrupt_anim.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, format="PNG")

    class MockIterator:
        def __init__(self, im):
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.count == 0:
                self.count += 1
                mock_frame = MagicMock()
                mock_frame.width = 100
                mock_frame.height = 100
                return mock_frame
            raise ValueError("Corrupt frame data")

    with patch("PIL.ImageSequence.Iterator", side_effect=MockIterator):
        err = _validate_raster_image_decodable(img_path)
        assert err is not None
        assert "Image could not be fully decoded" in err


def test_resize_image_for_vision_aspect_scale_to_max_dimension(tmp_path: Path):
    """Large dense document (e.g. 3539x2499) with max_dimension=1568 should scale directly to 1568x1107

    rather than halving multiple times down to illegible 884x624.
    """
    img_path = tmp_path / "document.jpg"
    orig_w, orig_h = 3539, 2499
    img = Image.new("RGB", (orig_w, orig_h), color="white")
    img.save(img_path, format="JPEG", quality=90)

    scale_out = {}
    data_uri = _resize_image_for_vision(
        img_path,
        max_dimension=1568,
        scale_out=scale_out,
    )

    assert data_uri.startswith("data:image/jpeg;base64,")
    b64_data = data_uri.split(",", 1)[1]
    decoded_bytes = base64.b64decode(b64_data)
    result_img = Image.open(io.BytesIO(decoded_bytes))

    # Verify scaled dimensions
    assert result_img.width == 1568
    # 2499 * (1568 / 3539) = 1107.15 -> 1107
    assert result_img.height == 1107

    # Verify scale_out dictionary
    assert scale_out.get("orig_width") == orig_w
    assert scale_out.get("orig_height") == orig_h
    assert scale_out.get("new_width") == 1568
    assert scale_out.get("new_height") == 1107


def test_resize_image_for_vision_height_dominant_scale(tmp_path: Path):
    """Tall vertical image scaled to max_dimension preserves aspect ratio based on height."""
    img_path = tmp_path / "tall.jpg"
    orig_w, orig_h = 1000, 3000
    img = Image.new("RGB", (orig_w, orig_h), color="gray")
    img.save(img_path, format="JPEG")

    scale_out = {}
    data_uri = _resize_image_for_vision(
        img_path,
        max_dimension=1500,
        scale_out=scale_out,
    )

    b64_data = data_uri.split(",", 1)[1]
    result_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))

    assert result_img.height == 1500
    assert result_img.width == 500
    assert scale_out.get("new_height") == 1500
    assert scale_out.get("new_width") == 500
