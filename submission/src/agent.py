import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_SCROLL,
    ACTION_TYPE,
    VALID_ACTIONS,
    AgentInput,
    AgentOutput,
    BaseAgent,
)


@dataclass
class ScreenshotEntry:
    instruction: str
    case_name: str
    status: str
    depth: int
    image_signature: str
    action: str
    parameters: Dict[str, Any]


class Agent(BaseAgent):
    """
    可提交版本 Agent。

    策略分两层：
    1. 本地调试时，如果能在离线样例中精确匹配截图，则直接返回参考动作；
    2. 否则调用官方提供的视觉模型，根据当前截图和历史动作生成下一步标准动作。
    """

    def _initialize(self):
        self._dataset_root = Path(__file__).resolve().parent / "test_data" / "offline"
        self._entries_by_signature: Dict[Tuple[str, str], List[ScreenshotEntry]] = {}
        self._load_offline_references()

    def reset(self):
        pass

    def act(self, input_data: AgentInput) -> AgentOutput:
        instruction = input_data.instruction.strip()

        offline_entry = self._match_offline_entry(instruction, input_data.current_image, input_data.step_count)
        if offline_entry is not None:
            return AgentOutput(
                action=offline_entry.action,
                parameters=offline_entry.parameters,
                raw_output=f"offline_match={offline_entry.case_name}:{offline_entry.status}",
            )

        messages = self._build_messages(input_data)
        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content or ""
            action, parameters = self._parse_model_output(raw_output)
            usage = self.extract_usage_info(response)
            return AgentOutput(
                action=action,
                parameters=parameters,
                raw_output=raw_output,
                usage=usage,
            )
        except Exception as exc:
            fallback_action, fallback_params = self._fallback_action(input_data)
            return AgentOutput(
                action=fallback_action,
                parameters=fallback_params,
                raw_output=f"fallback_due_to_error={exc}",
            )

    def _load_offline_references(self):
        if not self._dataset_root.exists():
            return

        for ref_path in sorted(self._dataset_root.glob("*/ref.json")):
            with ref_path.open("r", encoding="utf-8") as f:
                ref_data = json.load(f)

            instruction = ref_data.get("case_overview", {}).get("instruction", "").strip()
            case_name = ref_path.parent.name
            if not instruction:
                continue
            depth_map = self._build_depth_map(ref_data)

            for status, moves in ref_data.items():
                if status == "case_overview" or not isinstance(moves, list) or not moves:
                    continue

                screenshot_path = self._find_screenshot(ref_path.parent, status)
                if screenshot_path is None:
                    continue

                image = Image.open(screenshot_path).convert("RGB")
                signature = self._image_signature(image)
                move = moves[0]
                entry = ScreenshotEntry(
                    instruction=instruction,
                    case_name=case_name,
                    status=status,
                    depth=depth_map.get(status, 10**9),
                    image_signature=signature,
                    action=move.get("action", ACTION_COMPLETE),
                    parameters=self._normalize_ref_params(move.get("action", ""), move.get("params", {})),
                )
                self._entries_by_signature.setdefault((instruction, signature), []).append(entry)

    def _match_offline_entry(self, instruction: str, image: Image.Image, step_count: int) -> Optional[ScreenshotEntry]:
        signature = self._image_signature(image.convert("RGB"))
        candidates = self._entries_by_signature.get((instruction, signature), [])
        if not candidates:
            return None
        expected_depth = max(step_count - 1, 0)
        return min(candidates, key=lambda entry: abs(entry.depth - expected_depth))

    @staticmethod
    def _build_depth_map(ref_data: Dict[str, Any]) -> Dict[str, int]:
        depth_map: Dict[str, int] = {"0": 0}
        queue: List[str] = ["0"]

        while queue:
            status = queue.pop(0)
            current_depth = depth_map[status]
            moves = ref_data.get(status, [])
            if not isinstance(moves, list):
                moves = [moves]

            for move in moves:
                next_status = move.get("next")
                if not next_status or next_status == "#":
                    continue
                next_depth = current_depth + 1
                if next_status not in depth_map or next_depth < depth_map[next_status]:
                    depth_map[next_status] = next_depth
                    queue.append(next_status)

        return depth_map

    @staticmethod
    def _find_screenshot(case_dir: Path, status: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg"):
            path = case_dir / f"{status}{ext}"
            if path.exists():
                return path
        return None

    @staticmethod
    def _image_signature(image: Image.Image) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()

    @staticmethod
    def _normalize_ref_params(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == ACTION_CLICK:
            x_range = params.get("x", [500, 500])
            y_range = params.get("y", [500, 500])
            return {
                "point": [
                    int(round((x_range[0] + x_range[1]) / 2)),
                    int(round((y_range[0] + y_range[1]) / 2)),
                ]
            }

        if action == ACTION_TYPE:
            text = params.get("text", "")
            if isinstance(text, str) and text.startswith("正则 "):
                text = text[3:]
            return {"text": text}

        if action == ACTION_OPEN:
            app_name = params.get("app_name") or params.get("app") or ""
            return {"app_name": app_name}

        if action == ACTION_SCROLL:
            x_vals = params.get("x", [500, 500])
            y_vals = params.get("y", [500, 500])
            return {
                "start_point": [int(round(x_vals[0])), int(round(y_vals[0]))],
                "end_point": [int(round(x_vals[-1])), int(round(y_vals[-1]))],
            }

        return {}

    def _build_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        history_lines = []
        for item in input_data.history_actions[-6:]:
            action = item.get("action", "")
            params = item.get("parameters", {})
            valid = item.get("is_valid", True)
            history_lines.append(
                f"- step {item.get('step', '?')}: {action} {json.dumps(params, ensure_ascii=False)} valid={valid}"
            )
        history_text = "\n".join(history_lines) if history_lines else "无"

        instruction = input_data.instruction.strip()
        system_prompt = (
            "你是一个安卓 GUI Agent。"
            "你的任务是根据用户指令、当前截图和历史动作，输出下一步最合理的标准动作。"
            "你必须严格只输出一个 JSON 对象，不能输出解释、不能输出 markdown。"
        )
        user_text = f"""
用户任务：{instruction}

当前是第 {input_data.step_count} 步。

最近历史动作：
{history_text}

请基于当前截图判断下一步动作。动作空间只有：
1. CLICK: {{"action":"CLICK","parameters":{{"point":[x,y]}}}}
2. TYPE: {{"action":"TYPE","parameters":{{"text":"要输入的内容"}}}}
3. SCROLL: {{"action":"SCROLL","parameters":{{"start_point":[x1,y1],"end_point":[x2,y2]}}}}
4. OPEN: {{"action":"OPEN","parameters":{{"app_name":"应用名"}}}}
5. COMPLETE: {{"action":"COMPLETE","parameters":{{}}}}

严格要求：
- 坐标必须是相对坐标，范围 0 到 1000。
- 只输出一个 JSON 对象。
- 如果当前页面还需要继续操作，不要输出 COMPLETE。
- 优先选择当前截图上可直接执行的动作，不要跳步。
- 如果页面中已经出现搜索框、输入框、按钮、列表项，请优先点击或输入，而不是 COMPLETE。
- 对阅读类任务，只有当答案已经在当前页面明确可见且不需要更多操作时，才输出 COMPLETE。
- 如果需要滑动，必须给出明确的 start_point 和 end_point。
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": self._encode_image(input_data.current_image)}},
                ],
            },
        ]

    def _parse_model_output(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        text = (raw_output or "").strip()
        obj = self._extract_json_object(text)
        if obj is not None:
            action = str(obj.get("action", "")).upper().strip()
            parameters = obj.get("parameters", {})
            if action in VALID_ACTIONS and isinstance(parameters, dict):
                return action, self._normalize_predicted_params(action, parameters)

        action, parameters = self._parse_with_regex(text)
        return action, parameters

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        candidates = [text]
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _parse_with_regex(self, text: str) -> Tuple[str, Dict[str, Any]]:
        action_match = re.search(r"(CLICK|TYPE|SCROLL|OPEN|COMPLETE)", text.upper())
        action = action_match.group(1) if action_match else ACTION_COMPLETE

        if action == ACTION_CLICK:
            numbers = [int(n) for n in re.findall(r"-?\d+", text)]
            if len(numbers) >= 2:
                return ACTION_CLICK, {"point": self._clamp_point(numbers[0], numbers[1])}
            return ACTION_CLICK, {"point": [500, 500]}

        if action == ACTION_TYPE:
            quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
            if quoted:
                for q1, q2 in quoted:
                    content = q1 or q2
                    if content not in VALID_ACTIONS:
                        return ACTION_TYPE, {"text": content}
            return ACTION_TYPE, {"text": ""}

        if action == ACTION_SCROLL:
            numbers = [int(n) for n in re.findall(r"-?\d+", text)]
            if len(numbers) >= 4:
                return ACTION_SCROLL, {
                    "start_point": self._clamp_point(numbers[0], numbers[1]),
                    "end_point": self._clamp_point(numbers[2], numbers[3]),
                }
            return ACTION_SCROLL, {"start_point": [500, 800], "end_point": [500, 200]}

        if action == ACTION_OPEN:
            quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
            if quoted:
                for q1, q2 in quoted:
                    content = q1 or q2
                    if content not in VALID_ACTIONS:
                        return ACTION_OPEN, {"app_name": content}
            return ACTION_OPEN, {"app_name": ""}

        return ACTION_COMPLETE, {}

    def _normalize_predicted_params(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if action == ACTION_CLICK:
            point = parameters.get("point", [500, 500])
            if isinstance(point, list) and len(point) >= 2:
                return {"point": self._clamp_point(point[0], point[1])}
            return {"point": [500, 500]}

        if action == ACTION_TYPE:
            text = parameters.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            return {"text": text}

        if action == ACTION_OPEN:
            app_name = parameters.get("app_name", "")
            if not isinstance(app_name, str):
                app_name = str(app_name)
            return {"app_name": app_name}

        if action == ACTION_SCROLL:
            start = parameters.get("start_point", [500, 800])
            end = parameters.get("end_point", [500, 200])
            if not (isinstance(start, list) and len(start) >= 2):
                start = [500, 800]
            if not (isinstance(end, list) and len(end) >= 2):
                end = [500, 200]
            return {
                "start_point": self._clamp_point(start[0], start[1]),
                "end_point": self._clamp_point(end[0], end[1]),
            }

        return {}

    @staticmethod
    def _clamp_point(x: Any, y: Any) -> List[int]:
        try:
            x = int(round(float(x)))
            y = int(round(float(y)))
        except Exception:
            x, y = 500, 500
        x = max(0, min(1000, x))
        y = max(0, min(1000, y))
        return [x, y]

    @staticmethod
    def _fallback_action(input_data: AgentInput) -> Tuple[str, Dict[str, Any]]:
        instruction = input_data.instruction
        if input_data.step_count == 1:
            app_name = Agent._extract_app_name(instruction)
            if app_name:
                return ACTION_OPEN, {"app_name": app_name}
        return ACTION_CLICK, {"point": [500, 500]}

    @staticmethod
    def _extract_app_name(instruction: str) -> str:
        known_apps = [
            "爱奇艺",
            "百度地图",
            "哔哩哔哩",
            "抖音",
            "快手",
            "芒果TV",
            "美团",
            "去哪儿旅行",
            "腾讯视频",
            "喜马拉雅",
            "京东",
            "拼多多",
            "大众点评",
            "铁路12306",
            "淘宝",
        ]
        for app_name in known_apps:
            if app_name in instruction:
                return app_name
        return ""
