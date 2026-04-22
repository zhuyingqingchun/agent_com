from typing import Any, Dict


def make_initial_state() -> Dict[str, Any]:
    return {
        "instruction": "",
        "app_name": "",
        "task_type": "generic",
        "slots": {},
        "notes": [],
        "typed_texts": [],
        "last_action": "",
        "last_parameters": {},
        "repeat_action_count": 0,
        "screen_signatures": [],
        "screen_features": [],
        "last_image": None,
        "phase": "launch",
        "milestone": "launch_app",
        "phase_retry_count": 0,
        "page_type": "unknown",
        "last_history_len": 0,
        "search_box_clicked": False,
        "input_activated": False,
        "complete_ready": False,
        "submit_ready": False,
    }


def reset_task_state(
    state: Dict[str, Any],
    *,
    instruction: str,
    app_name: str,
    task_type: str,
    slots: Dict[str, str],
    launch_milestone: str,
) -> Dict[str, Any]:
    state.update(
        {
            "instruction": instruction,
            "app_name": app_name,
            "task_type": task_type,
            "slots": slots,
            "notes": [],
            "typed_texts": [],
            "last_action": "",
            "last_parameters": {},
            "repeat_action_count": 0,
            "screen_signatures": [],
            "screen_features": [],
            "last_image": None,
            "phase": "launch",
            "milestone": launch_milestone,
            "phase_retry_count": 0,
            "page_type": "unknown",
            "last_history_len": 0,
            "search_box_clicked": False,
            "input_activated": False,
            "complete_ready": False,
            "submit_ready": False,
        }
    )
    return state
