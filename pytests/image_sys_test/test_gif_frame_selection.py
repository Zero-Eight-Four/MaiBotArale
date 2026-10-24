from io import BytesIO

from PIL import Image
import pytest

from src.common.utils.utils_image import ImageUtils


def _make_gif(levels: list[int]) -> bytes:
    frames = [Image.new("RGB", (20, 20), (level, level, level)) for level in levels]
    buffer = BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return buffer.getvalue()


@pytest.mark.parametrize("levels", [[0, 255, 0], [255, 0, 255]])
def test_gif_keeps_distinct_frames_in_order(levels: list[int]) -> None:
    output = ImageUtils.gif_2_static_image(_make_gif(levels))
    with Image.open(BytesIO(output)) as image:
        assert image.format == "JPEG"
        assert image.size == (600, 200)
        for index, level in enumerate(levels):
            pixel = image.getpixel((index * 200 + 100, 100))
            assert all(abs(channel - level) <= 2 for channel in pixel)


def test_gif_compares_against_last_selected_frame() -> None:
    # Each adjacent change is below the threshold, but the total change exceeds it.
    output = ImageUtils.gif_2_static_image(_make_gif([0, 20, 40]))
    with Image.open(BytesIO(output)) as image:
        assert image.size == (400, 200)
        assert image.getpixel((100, 100)) == (0, 0, 0)
        assert all(abs(channel - 40) <= 2 for channel in image.getpixel((300, 100)))


def test_gif_respects_frame_limit() -> None:
    output = ImageUtils.gif_2_static_image(_make_gif([0, 255, 0, 255]), max_frames=2)
    with Image.open(BytesIO(output)) as image:
        assert image.size == (400, 200)
