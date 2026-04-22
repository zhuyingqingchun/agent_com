"""
动作后处理模块 - 处理模型输出的动作

核心设计原则：
- 代码不做强制动作替换，只做解析、格式化、状态跟踪
- 模型输出什么就传什么，除非约束系统（调模型重判）介入
- 约束系统是唯一的"矫正"入口，且矫正方式也是调用模型让模型自己决定
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE
from utils.agent_regions import clamp_point, refine_click_point, region_center, same_click


class ActionProcessor:
    """动作处理器 - 轻量化，不做强制干预"""

    def __init__(
        self,
        state: Dict[str, Any],
        candidate_regions: Dict[str, list],
        phase_transitions: Dict[str, str],
        call_api: Optional[Callable] = None,
        current_image: Any = None,
        encode_image: Optional[Callable] = None,
        make_grid_image: Optional[Callable] = None,
        grid_cols: int = 4,
        grid_rows: int = 6,
        include_grid_image: bool = True,
    ):
        self.state = state
        self.candidate_regions = candidate_regions
        self.phase_transitions = phase_transitions
        self._call_api = call_api
        self._current_image = current_image
        self._encode_image = encode_image
        self._make_grid_image = make_grid_image
        self._grid_cols = grid_cols
        self._grid_rows = grid_rows
        self._include_grid_image = include_grid_image

    def process(
        self,
        action: str,
        parameters: Dict[str, Any],
        input_data: Any,
        page_stuck: bool,
        raw_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """处理动作：格式化参数 + 更新状态 + 约束检查（仅模型驱动）"""
        if action == ACTION_OPEN:
            return self._process_open(parameters, input_data)

        if action == ACTION_TYPE:
            return self._process_type(parameters)

        if action == ACTION_COMPLETE:
            return self._process_complete(input_data, page_stuck, raw_text)

        if action == ACTION_CLICK:
            return self._process_click(parameters, input_data, page_stuck, raw_text)

        if action == ACTION_SCROLL:
            return self._process_scroll(parameters)

        return action, parameters if isinstance(parameters, dict) else {}

    def _process_open(self, parameters: Dict[str, Any], input_data: Any) -> Tuple[str, Dict[str, Any]]:
        app_name = parameters.get("app_name", "") if isinstance(parameters, dict) else ""
        if not app_name:
            from utils.agent_rules import extract_app_name
            app_name = self.state.get("app_name") or extract_app_name(input_data.instruction)
        self.state["phase"] = "home"
        return ACTION_OPEN, {"app_name": app_name}

    def _process_type(self, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        text = parameters.get("text", "") if isinstance(parameters, dict) else ""
        text = text.strip()
        if not text:
            slots = self.state.get("slots", {})
            text = slots.get("query") or slots.get("destination") or slots.get("keyword") or ""
        typed_texts = self.state.get("typed_texts", [])
        if text and text not in typed_texts:
            typed_texts.append(text)
            self.state["typed_texts"] = typed_texts
        return ACTION_TYPE, {"text": text}

    def _process_complete(self, input_data: Any, page_stuck: bool, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        override = self._apply_confirm_complete_constraint(input_data, raw_text)
        if override:
            return override
        self.state["phase"] = "complete"
        self.state["complete_ready"] = True
        return ACTION_COMPLETE, {}

    def _process_click(
        self,
        parameters: Dict[str, Any],
        input_data: Any,
        page_stuck: bool,
        raw_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        point = parameters.get("point", [500, 500]) if isinstance(parameters, dict) else [500, 500]
        region = parameters.get("_candidate_region", "") if isinstance(parameters, dict) else ""
        phase = self.state.get("phase", "launch")

        point = clamp_point(point[0], point[1]) if isinstance(point, list) and len(point) >= 2 else region_center(region, self.candidate_regions)
        point = refine_click_point(point, region, self.candidate_regions)

        override_result = self._apply_detail_confirm_constraint(point, region, phase, input_data)
        if override_result is not None:
            return override_result

        return ACTION_CLICK, {"point": point}

    def _process_scroll(self, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        start = parameters.get("start_point", [500, 800]) if isinstance(parameters, dict) else [500, 800]
        end = parameters.get("end_point", [500, 200]) if isinstance(parameters, dict) else [500, 200]
        return ACTION_SCROLL, {
            "start_point": clamp_point(start[0], start[1]),
            "end_point": clamp_point(end[0], end[1]),
        }

    def _append_note(self, note: str):
        notes = self.state.get("notes", [])
        notes.append(note)
        self.state["notes"] = notes[-20:]

    # ==========================================
    #   约束1：detail/confirm 阶段顶部点击矫正（模型驱动）
    # ==========================================

    def _apply_detail_confirm_constraint(
        self,
        point: list,
        region: str,
        phase: str,
        input_data: Any,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        if phase not in {"detail", "confirm"}:
            return None
        if not self._call_api or not self._current_image:
            return None
        y_coord = point[1] if isinstance(point, list) and len(point) >= 2 else 500
        if y_coord >= 200:
            return None

        typed_texts = self.state.get("typed_texts", [])
        if not typed_texts:
            return None

        self._append_note(f"约束1触发：{phase}阶段检测到顶部点击({point})，调用模型重判。")
        result = self._call_phase_corrector(input_data, point)
        return result

    def _call_phase_corrector(self, input_data: Any, original_point: list) -> Tuple[str, Dict[str, Any]]:
        import json
        phase = self.state.get("phase", "unknown")
        typed_texts = self.state.get("typed_texts", [])
        instruction = getattr(input_data, "instruction", "") or ""
        step_count = getattr(input_data, "step_count", 0)

        history_items = getattr(input_data, "history_actions", []) or []
        history_lines = [
            f"- step {h.get('step', '?')}: {h.get('action', '')} {json.dumps(h.get('parameters', {}), ensure_ascii=False)}"
            for h in history_items[-5:]
        ]
        history_str = "\n".join(history_lines) if history_lines else "- 无"

        user_prompt = f"""【任务指令】
{instruction}

【当前阶段】{phase}（已执行 {step_count} 步）
【已输入过的文本】{typed_texts if typed_texts else '无'}

【历史动作】
{history_str}

【问题】
主模型在 {phase} 阶段输出了一个靠近屏幕顶部的 CLICK({original_point})。
但这在 {phase} 阶段通常不合理——此时你应该关注页面的中下部内容。

请仔细观察截图，结合任务指令判断：
1. 任务是否还需要你输入某段文字？（比如评论内容、留言、回复等）
2. 如果需要输入文字，输出 TYPE 和具体文本内容
3. 如果不需要输入文字，输出一个更合理的 CLICK 坐标（指向中下部的目标元素）

只输出 JSON："""

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if self._current_image is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(self._current_image)},
            })
        if self._include_grid_image and self._make_grid_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(
                    self._make_grid_image(self._current_image, self._grid_cols, self._grid_rows))},
            })

        messages = [
            {"role": "system", "content": (
                "你是 GUI 动作校正器。用户的主模型在 detail/confirm 阶段给出了一个可疑的顶部点击。"
                "你需要根据截图和任务指令，判断正确的下一步动作。"
                "只输出一个 JSON 对象。"
            )},
            {"role": "user", "content": content},
        ]

        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content or ""
            from utils.agent_parser import OutputParser
            parser = OutputParser(self.candidate_regions)
            action, params = parser.parse(raw_output)
            if action in {ACTION_CLICK, ACTION_TYPE, ACTION_SCROLL, ACTION_COMPLETE}:
                self._append_note(f"约束1-模型重判结果: {action} {params}")
                return action, params
        except Exception as exc:
            self._append_note(f"约束1-模型调用失败: {exc}，回退到中下部点击。")

        center_point = region_center("CENTER_PANEL", self.candidate_regions)
        return ACTION_CLICK, {"point": center_point}

    # ==========================================
    #   约束2：confirm 阶段过早 COMPLETE 矫正（模型驱动）
    # ==========================================

    def _apply_confirm_complete_constraint(
        self,
        input_data: Any,
        raw_text: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        phase = self.state.get("phase", "launch")
        if phase != "confirm":
            return None
        if not self._call_api or not self._current_image:
            return None

        typed_texts = self.state.get("typed_texts", [])
        if not typed_texts:
            return None

        last_action = self.state.get("last_action", "")
        if last_action != ACTION_TYPE:
            return None

        step_count = getattr(input_data, "step_count", 0)
        if step_count >= 20:
            return None

        last_action = self.state.get("last_action", "")
        last_parameters = self.state.get("last_parameters", {})
        last_point = last_parameters.get("point", []) if isinstance(last_parameters, dict) else []

        already_clicked_bottom = (
            last_action == ACTION_CLICK
            and isinstance(last_point, list)
            and len(last_point) >= 2
            and last_point[1] >= 700
        )
        if already_clicked_bottom:
            return None

        self._append_note("约束2触发：confirm阶段有输入记录但未确认执行，调用模型重判。")
        result = self._call_completion_checker(input_data, raw_text)
        return result

    def _call_completion_checker(self, input_data: Any, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        import json
        typed_texts = self.state.get("typed_texts", [])
        instruction = getattr(input_data, "instruction", "") or ""
        step_count = getattr(input_data, "step_count", 0)

        history_items = getattr(input_data, "history_actions", []) or []
        history_lines = [
            f"- step {h.get('step', '?')}: {h.get('action', '')} {json.dumps(h.get('parameters', {}), ensure_ascii=False)}"
            for h in history_items[-5:]
        ]
        history_str = "\n".join(history_lines) if history_lines else "- 无"

        user_prompt = f"""【任务指令】
{instruction}

【当前阶段】confirm（已执行 {step_count} 步）
【已输入过的文本】{typed_texts}

【历史动作】
{history_str}

【问题】
主模型在 confirm 阶段输出了 COMPLETE。
但你已经输入过文本内容（{typed_texts}），且上一步还没有点击发送/提交/确认按钮。

请仔细观察截图：
1. 截图中是否有"发送"、"发布"、"提交"、"确认"、"评论"之类的按钮？
2. 如果有，输出 CLICK 和那个按钮的大致坐标
3. 如果截图确实显示任务已经完成（比如评论已经发出去了），才输出 COMPLETE

只输出 JSON："""

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if self._current_image is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(self._current_image)},
            })
        if self._include_grid_image and self._make_grid_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(
                    self._make_grid_image(self._current_image, self._grid_cols, self._grid_rows))},
            })

        messages = [
            {"role": "system", "content": (
                "你是 GUI 动作校正器。主模型在 confirm 阶段过早输出了 COMPLETE。"
                "你需要根据截图和任务指令判断：任务真的完成了还是还差一步？"
                "只输出一个 JSON 对象。"
            )},
            {"role": "user", "content": content},
        ]

        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content or ""
            from utils.agent_parser import OutputParser
            parser = OutputParser(self.candidate_regions)
            action, params = parser.parse(raw_output)
            if action in {ACTION_CLICK, ACTION_TYPE, ACTION_SCROLL, ACTION_COMPLETE}:
                self._append_note(f"约束2-模型重判结果: {action} {params}")
                return action, params
        except Exception as exc:
            self._append_note(f"约束2-模型调用失败: {exc}，保持原COMPLETE。")

        return ACTION_COMPLETE, {}
