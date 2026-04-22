from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ScreenshotEntry:
    instruction: str
    case_name: str
    status: str
    depth: int
    image_signature: str
    action: str
    parameters: Dict[str, Any]


KNOWN_APPS = [
    "美团",
    "抖音",
    "百度地图",
    "爱奇艺",
    "哔哩哔哩",
    "快手",
    "芒果TV",
    "腾讯视频",
    "喜马拉雅",
    "去哪儿旅行",
]

CANDIDATE_REGIONS = {
    "TOP_BAR": [500, 110],
    "TOP_SEARCH_BOX": [500, 110],
    "TOP_LEFT": [180, 110],
    "TOP_LEFT_ICON": [88, 110],
    "TOP_RIGHT": [860, 110],
    "TOP_RIGHT_ICON": [930, 110],
    "TOP_RIGHT_SMALL": [880, 110],
    "CENTER_PANEL": [500, 430],
    "MID_LIST": [500, 560],
    "LEFT_PANEL": [240, 520],
    "RIGHT_PANEL": [760, 520],
    "BOTTOM_BAR": [500, 900],
    "BOTTOM_LEFT": [220, 900],
    "BOTTOM_RIGHT": [780, 900],
}

REGION_CROP_RANGES = {
    "TOP_BAR": (0.00, 0.00, 1.00, 0.22),
    "TOP_SEARCH_BOX": (0.08, 0.02, 0.92, 0.18),
    "TOP_LEFT_ICON": (0.00, 0.00, 0.20, 0.18),
    "TOP_RIGHT_ICON": (0.80, 0.00, 1.00, 0.18),
    "TOP_RIGHT_SMALL": (0.72, 0.00, 0.96, 0.18),
    "CENTER_PANEL": (0.08, 0.18, 0.92, 0.72),
    "MID_LIST": (0.05, 0.22, 0.95, 0.86),
    "BOTTOM_BAR": (0.00, 0.78, 1.00, 1.00),
    "LEFT_PANEL": (0.00, 0.18, 0.55, 0.82),
    "RIGHT_PANEL": (0.45, 0.18, 1.00, 0.82),
}

PHASE_TRANSITIONS = {
    "launch": "launch_app",
    "home": "find_entry",
    "search_entry": "open_search",
    "search_input": "type_keyword",
    "submit_search": "submit_or_confirm",
    "results": "browse_results",
    "detail": "inspect_detail",
    "confirm": "final_confirmation",
    "complete": "done",
}
