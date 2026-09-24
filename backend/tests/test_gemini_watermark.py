import pytest

from backend.explainers import Style
from backend.pipeline.gemini_watermark import WatermarkRegion, repair_plan


@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (720, 1280)])
def test_region_scales_with_source_and_keeps_interpolation_border(width, height):
    plan = repair_plan(width, height)
    box = plan["region_pixels"]
    assert abs(box["x"] / width - 0.875) < 1 / width
    assert abs(box["y"] / height - 0.7778) < 1 / height
    assert 0 < box["x"] < box["x"] + box["width"] < width
    assert 0 < box["y"] < box["y"] + box["height"] < height
    assert plan["source_dimensions"] == [width, height]


@pytest.mark.parametrize("region", [
    {"x": 99}, {"y": 99}, {"width": 0}, {"height": -1}, {"x": float("nan")},
    {"x": "1:show=1"}, {"height": float("inf")}, {"x": 93.75},
])
def test_rejects_invalid_regions(region):
    with pytest.raises(ValueError):
        WatermarkRegion.model_validate(region)


def test_rejects_region_without_enough_source_pixels():
    with pytest.raises(ValueError, match="too small"):
        repair_plan(32, 18)


def test_legacy_style_enables_repair_and_explicit_opt_out_survives():
    assert Style.model_validate({"motion_mode": "gemini"}).gemini_watermark_removal is True
    assert Style(gemini_watermark_removal=False).model_dump()["gemini_watermark_removal"] is False
