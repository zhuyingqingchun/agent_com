import hashlib
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw


def image_signature(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def screen_feature(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)


def clamp_point(x: Any, y: Any) -> List[int]:
    try:
        x = int(round(float(x)))
        y = int(round(float(y)))
    except Exception:
        x, y = 500, 500
    x = max(0, min(1000, x))
    y = max(0, min(1000, y))
    return [x, y]


def make_grid_image(image: Image.Image, grid_cols: int, grid_rows: int) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for col in range(1, grid_cols):
        x = int(round(w * col / grid_cols))
        draw.line((x, 0, x, h), fill=(255, 64, 64), width=3)

    for row in range(1, grid_rows):
        y = int(round(h * row / grid_rows))
        draw.line((0, y, w, y), fill=(255, 64, 64), width=3)

    cell_w = w / grid_cols
    cell_h = h / grid_rows
    for row in range(grid_rows):
        for col in range(grid_cols):
            label = f"{chr(ord('A') + row)}{col + 1}"
            x = int(col * cell_w + 10)
            y = int(row * cell_h + 10)
            draw.rectangle((x, y, x + 56, y + 24), fill=(0, 0, 0))
            draw.text((x + 4, y + 4), label, fill=(255, 255, 0))
    return img


def annotate_crop(image: Image.Image, region_name: str) -> Image.Image:
    crop = image.copy()
    draw = ImageDraw.Draw(crop)
    draw.rectangle((0, 0, min(crop.size[0] - 1, 170), 28), fill=(0, 0, 0))
    draw.text((6, 6), region_name, fill=(255, 255, 0))
    return crop


def make_region_crops(image: Image.Image, region_crop_ranges: Dict[str, tuple]) -> Dict[str, Image.Image]:
    img = image.convert("RGB")
    w, h = img.size
    crops: Dict[str, Image.Image] = {}
    for region_name, (x1r, y1r, x2r, y2r) in region_crop_ranges.items():
        x1 = int(round(w * x1r))
        y1 = int(round(h * y1r))
        x2 = int(round(w * x2r))
        y2 = int(round(h * y2r))
        crop = img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
        crops[region_name] = annotate_crop(crop, region_name)
    return crops


def region_center(region_name: str, candidate_regions: Dict[str, List[int]]) -> List[int]:
    return list(candidate_regions.get(region_name or "", [500, 500]))


def snap_point_to_region(point: List[int], region_name: str, candidate_regions: Dict[str, List[int]]) -> List[int]:
    region_name = (region_name or "").upper().strip()
    if region_name not in candidate_regions:
        return point
    center = region_center(region_name, candidate_regions)
    px, py = point
    cx, cy = center
    px = int(round((px * 0.6) + (cx * 0.4)))
    py = int(round((py * 0.6) + (cy * 0.4)))
    return clamp_point(px, py)


def same_click(last_parameters: Dict[str, Any], point: List[int]) -> bool:
    last_point = last_parameters.get("point", []) if isinstance(last_parameters, dict) else []
    return bool(isinstance(last_point, list) and len(last_point) >= 2 and last_point == point)


def region_candidate_points(region_name: str, candidate_regions: Dict[str, List[int]]) -> List[List[int]]:
    region_name = (region_name or "").upper().strip()
    candidates = {
        "TOP_SEARCH_BOX": [[500, 72], [440, 72], [560, 72], [360, 82], [640, 82]],
        "TOP_RIGHT_ICON": [[900, 48], [870, 52], [930, 52], [840, 60]],
        "TOP_RIGHT_SMALL": [[870, 48], [845, 52], [900, 52], [825, 60]],
        "TOP_LEFT_ICON": [[88, 52], [120, 56], [70, 56], [150, 62]],
        "TOP_BAR": [[500, 72], [420, 72], [580, 72]],
        "MID_LIST": [[500, 560], [500, 500], [500, 620], [420, 560], [580, 560]],
        "RIGHT_PANEL": [[760, 520], [820, 480], [860, 420]],
        "LEFT_PANEL": [[240, 520], [180, 480], [140, 420]],
        "BOTTOM_BAR": [[500, 900], [420, 900], [580, 900]],
        "BOTTOM_RIGHT": [[880, 916], [840, 900], [930, 900], [900, 860]],
        "BOTTOM_LEFT": [[220, 900], [160, 860], [100, 820]],
        "CENTER_PANEL": [[500, 430], [500, 500], [500, 360]],
    }
    return [clamp_point(x, y) for x, y in candidates.get(region_name, [region_center(region_name, candidate_regions)])]


def clamp_point_to_region_band(point: List[int], region_name: str) -> List[int]:
    region_name = (region_name or "").upper().strip()
    x, y = point
    safe_boxes = {
        "TOP_RIGHT_ICON": ((820, 950), (35, 85)),
        "TOP_RIGHT_SMALL": ((800, 920), (35, 85)),
        "TOP_SEARCH_BOX": ((160, 840), (45, 110)),
        "TOP_LEFT_ICON": ((40, 220), (35, 90)),
        "BOTTOM_RIGHT": ((760, 980), (820, 960)),
        "BOTTOM_BAR": ((160, 840), (820, 960)),
    }
    if region_name not in safe_boxes:
        return point
    (x1, x2), (y1, y2) = safe_boxes[region_name]
    return [max(x1, min(x2, x)), max(y1, min(y2, y))]


def refine_click_point(
    point: List[int],
    region_name: str,
    candidate_regions: Dict[str, List[int]],
) -> List[int]:
    region_name = (region_name or "").upper().strip()
    if region_name not in candidate_regions:
        return point
    snapped = snap_point_to_region(point, region_name, candidate_regions)
    snapped = clamp_point_to_region_band(snapped, region_name)
    candidates = region_candidate_points(region_name, candidate_regions)
    if region_name in {"TOP_SEARCH_BOX", "TOP_RIGHT_ICON", "TOP_RIGHT_SMALL", "TOP_LEFT_ICON"}:
        scored = sorted(
            candidates,
            key=lambda p: abs(p[0] - snapped[0]) + abs(p[1] - snapped[1]),
        )
        return clamp_point_to_region_band(scored[0], region_name)
    return clamp_point_to_region_band(snapped, region_name)
