import json
from pathlib import Path
from typing import Any, Dict, List

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE


def load_app_memory(memory_path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not memory_path.exists():
        return {}
    try:
        with memory_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def persist_app_memory(memory_path: Path, app_memory: Dict[str, Dict[str, List[str]]]) -> None:
    try:
        with memory_path.open("w", encoding="utf-8") as f:
            json.dump(app_memory, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def store_app_memory(
    app_memory: Dict[str, Dict[str, List[str]]],
    memory_path: Path,
    app_name: str,
    task_type: str,
    note: str,
) -> None:
    app_name = app_name or "UNKNOWN_APP"
    bucket = app_memory.setdefault(app_name, {})
    notes = bucket.setdefault(task_type, [])
    if note not in notes:
        notes.append(note)
    bucket[task_type] = notes[-10:]
    persist_app_memory(memory_path, app_memory)


def retrieve_app_memory(
    app_memory: Dict[str, Dict[str, List[str]]],
    app_name: str,
    task_type: str,
) -> str:
    if not app_name:
        return ""
    bucket = app_memory.get(app_name, {})
    notes = list(bucket.get(task_type, [])) + list(bucket.get("generic", []))
    notes = notes[-5:]
    return "；".join(notes)


def summarize_valid_action(item: Dict[str, Any]) -> str:
    action = item.get("action", "")
    params = item.get("parameters", {}) or {}
    if action == ACTION_CLICK:
        point = params.get("point", [])
        if isinstance(point, list) and len(point) >= 2:
            x, y = point[:2]
            if y <= 180:
                return "顶部区域点击常可打开搜索入口；右上小控件常用于搜索、关闭或确认。"
            if y >= 820:
                return "底部区域点击常用于确认、切换 Tab 或关闭次级弹窗。"
            return "中部列表点击常用于进入结果项或详情页。"
    if action == ACTION_TYPE and params.get("text"):
        return "进入输入阶段后，直接输入 query 通常有效。"
    if action == ACTION_SCROLL:
        return "结果列表阶段可通过上滑继续探索候选项。"
    if action == ACTION_OPEN and params.get("app_name"):
        return f"任务首步优先直接 OPEN {params.get('app_name')}。"
    if action == ACTION_COMPLETE:
        return "只有目标结果在当前页明确可见时才 COMPLETE。"
    return ""
