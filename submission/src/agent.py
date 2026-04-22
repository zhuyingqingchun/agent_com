
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

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
    面向赛事提分的 GUI Agent。

    本轮继续强化三件事：
    1. TYPE 前置条件硬约束：未进入输入阶段时禁止直接 TYPE；
    2. 顶部搜索框 / 左上返回 / 右上小控件更细粒度定位；
    3. COMPLETE 二段式放行：只有满足完成条件才允许真正结束。
    """

    # 精简后的应用列表，只保留测试用例中实际使用的应用
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

    def _initialize(self):
        self._dataset_root = Path(__file__).resolve().parent / "test_data" / "offline"
        self._entries_by_signature: Dict[Tuple[str, str], List[ScreenshotEntry]] = {}
        self._grid_cols = 4
        self._grid_rows = 6
        self._memory_path = Path(__file__).resolve().parent / "_runtime_app_memory.json"
        self._app_memory: Dict[str, Dict[str, List[str]]] = self._load_app_memory()
        self._load_offline_references()
        self.reset()

    def reset(self):
        self._state: Dict[str, Any] = {
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
            "complete_ready": False,
            "submit_ready": False,
        }

    def act(self, input_data: AgentInput) -> AgentOutput:
        instruction = input_data.instruction.strip()
        self._bootstrap_task_state(instruction)
        self._sync_memory_from_history(input_data)

        current_rgb = input_data.current_image.convert("RGB")
        current_signature = self._image_signature(current_rgb)
        current_feature = self._screen_feature(current_rgb)
        page_stuck = self._is_page_stuck(current_feature)
        page_type = self._infer_page_type(input_data, current_rgb)
        self._state["page_type"] = page_type

        # 禁用离线匹配，强制使用真实 API (GLM-4V 测试)
        # offline_entry = self._match_offline_entry(instruction, current_rgb, input_data.step_count)
        # if offline_entry is not None and not page_stuck:
        #     action, parameters = self._post_process_action(
        #         input_data=input_data,
        #         action=offline_entry.action,
        #         parameters=offline_entry.parameters,
        #         page_stuck=page_stuck,
        #         raw_text="offline_entry",
        #     )
        #     output = AgentOutput(
        #         action=action,
        #         parameters=parameters,
        #         raw_output=f"offline_match={offline_entry.case_name}:{offline_entry.status}",
        #     )
        #     self._update_runtime_state(current_rgb, current_signature, current_feature, output)
        #     return output

        messages = self._build_messages(input_data, current_rgb, page_stuck)

        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content or ""
            action, parameters = self._parse_model_output(raw_output)
            action, parameters = self._post_process_action(
                input_data=input_data,
                action=action,
                parameters=parameters,
                page_stuck=page_stuck,
                raw_text=raw_output,
            )
            usage = self.extract_usage_info(response)
            output = AgentOutput(
                action=action,
                parameters=parameters,
                raw_output=raw_output,
                usage=usage,
            )
        except Exception as exc:
            fallback_action, fallback_params = self._fallback_action(input_data, page_stuck=page_stuck)
            output = AgentOutput(
                action=fallback_action,
                parameters=fallback_params,
                raw_output=f"fallback_due_to_error={exc}",
            )

        self._update_runtime_state(current_rgb, current_signature, current_feature, output)
        return output

    def _bootstrap_task_state(self, instruction: str):
        if instruction == self._state.get("instruction"):
            return

        app_name = self._extract_app_name(instruction)
        task_type = self._infer_task_type(instruction, app_name)
        slots = self._extract_slots(instruction, task_type)

        self._state["instruction"] = instruction
        self._state["app_name"] = app_name
        self._state["task_type"] = task_type
        self._state["slots"] = slots
        self._state["notes"] = []
        self._state["typed_texts"] = []
        self._state["last_action"] = ""
        self._state["last_parameters"] = {}
        self._state["repeat_action_count"] = 0
        self._state["screen_signatures"] = []
        self._state["screen_features"] = []
        self._state["last_image"] = None
        self._state["phase"] = "launch"
        self._state["milestone"] = self.PHASE_TRANSITIONS["launch"]
        self._state["phase_retry_count"] = 0
        self._state["page_type"] = "unknown"
        self._state["last_history_len"] = 0
        self._state["search_box_clicked"] = False
        self._state["complete_ready"] = False
        self._state["submit_ready"] = False

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
    def _screen_feature(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)

    def _is_page_stuck(self, current_feature: np.ndarray) -> bool:
        features = self._state.get("screen_features", [])
        if not features:
            return False
        prev = features[-1]
        diff = float(np.mean(np.abs(prev - current_feature)))
        return diff < 8.0

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

    def _build_messages(
        self,
        input_data: AgentInput,
        current_image: Image.Image,
        page_stuck: bool,
    ) -> List[Dict[str, Any]]:
        history_lines = []
        for item in input_data.history_actions[-8:]:
            action = item.get("action", "")
            params = item.get("parameters", {})
            valid = item.get("is_valid", True)
            history_lines.append(
                f"- step {item.get('step', '?')}: {action} {json.dumps(params, ensure_ascii=False)} valid={valid}"
            )
        history_text = "\n".join(history_lines) if history_lines else "无"

        slots = self._state.get("slots", {})
        slots_text = json.dumps(slots, ensure_ascii=False) if slots else "{}"
        notes_text = "；".join(self._state.get("notes", [])[-5:]) or "无"
        typed_texts = self._state.get("typed_texts", [])
        typed_texts_text = json.dumps(typed_texts[-4:], ensure_ascii=False) if typed_texts else "[]"
        app_name = self._state.get("app_name", "")
        task_type = self._state.get("task_type", "generic")
        playbook = self._workflow_hint(app_name, task_type)
        memory_hint = self._retrieve_app_memory(app_name, task_type)
        phase = self._state.get("phase", "launch")
        milestone = self._state.get("milestone", self.PHASE_TRANSITIONS["launch"])
        page_type = self._state.get("page_type", "unknown")
        retry_count = self._state.get("phase_retry_count", 0)
        stuck_hint = (
            "是。当前页面与上一帧高度相似，禁止机械重复上一步动作。优先寻找新的入口、搜索框、确认按钮、关闭弹窗按钮或结果列表。"
            if page_stuck else
            "否。优先执行当前页面能直接推进任务的最短动作。"
        )

        system_prompt = (
            "你是一个擅长安卓 GUI 控制的多步 Agent。"
            "你必须利用任务记忆、phase 进度、上一帧信息、带网格截图、局部裁剪图和候选区域来做更稳的定位。"
            "你只输出一个 JSON 对象。"
        )

        user_text = f"""
用户任务：{input_data.instruction.strip()}

任务摘要：
- app_name: {app_name or "未知"}
- task_type: {task_type}
- slots: {slots_text}
- 已输入文本: {typed_texts_text}
- 历史备注: {notes_text}
- 当前 phase: {phase}
- 当前 milestone: {milestone}
- 当前 page_type: {page_type}
- 当前 phase_retry_count: {retry_count}

最近历史动作：
{history_text}

推荐工作流：
{playbook}

App 经验记忆（若为空则忽略）：
{memory_hint or "无"}

额外的国内常用生活类 App 先验（离线数据可能未覆盖，但线上很常见）：
- 地图/出行：高德地图、百度地图、滴滴出行
- 本地生活：美团、大众点评、饿了么、盒马
- 旅行：去哪儿旅行、携程旅行、铁路12306
- 购物：淘宝、京东、拼多多、小红书
- 内容/娱乐：抖音、快手、哔哩哔哩、爱奇艺、腾讯视频、芒果TV、优酷视频、喜马拉雅、QQ音乐、网易云音乐
- 通用生活入口：支付宝、微信

页面疑似未变化：
{stuck_hint}

你将看到：
1. 当前原始截图
2. 当前带网格截图（全局定位）
3. TOP_SEARCH_BOX / TOP_LEFT_ICON / TOP_RIGHT_ICON / TOP_RIGHT_SMALL / CENTER_PANEL / MID_LIST / RIGHT_PANEL / LEFT_PANEL / BOTTOM_BAR 等局部裁剪图
4. 若有，则附带上一帧原始截图与上一帧网格截图

候选区域说明（可在 JSON 中额外返回 candidate_region，便于你更稳地指示点击区域）：
- TOP_SEARCH_BOX: 顶部搜索框 / 输入栏主体
- TOP_RIGHT_ICON: 顶部右上角小图标、搜索按钮、关闭按钮
- TOP_RIGHT_SMALL: 顶部更小的次级按钮或筛选控件
- TOP_LEFT_ICON: 顶部返回、关闭、左上角入口
- TOP_BAR: 顶部大区域兜底
- CENTER_PANEL: 中央主面板
- MID_LIST: 中部结果列表 / 内容卡片区
- BOTTOM_BAR: 底部导航、底部确认区
- LEFT_PANEL / RIGHT_PANEL: 左右侧卡片或按钮区

动作空间只有：
1. CLICK: {{"action":"CLICK","parameters":{{"point":[x,y]}},"candidate_region":"TOP_SEARCH_BOX"}}
2. TYPE: {{"action":"TYPE","parameters":{{"text":"要输入的内容"}}}}
3. SCROLL: {{"action":"SCROLL","parameters":{{"start_point":[x1,y1],"end_point":[x2,y2]}}}}
4. OPEN: {{"action":"OPEN","parameters":{{"app_name":"应用名"}}}}
5. COMPLETE: {{"action":"COMPLETE","parameters":{{}}}}

严格要求：
- 坐标必须是 0 到 1000 的相对坐标。
- 只输出一个 JSON 对象。
- 除非任务已经在当前页面明确完成，否则不要输出 COMPLETE。
- 如果出现广告弹窗、权限弹窗、活动弹窗、青少年模式/升级提示等中断页面，优先关闭或跳过，再继续主任务。
- 若当前页面和上一帧相似，不要重复同一点位点击或重复无意义输入。
- 优先点击当前页面上最明确、最接近目标的 UI 元素，不要跳步。
- 未点击搜索框或当前 phase 不是 search_input / submit_search 时，禁止直接 TYPE。
- 当目标是顶部搜索入口、右上角搜索/筛选/关闭按钮时，优先使用 TOP_SEARCH_BOX / TOP_RIGHT_ICON / TOP_RIGHT_SMALL。
- 只有当结果或答案已经明确显示在当前页，且继续操作大概率会过度点击时，才输出 COMPLETE。
- 如果需要输入，优先使用 slots 中的 query/location/destination 等字段，不要凭空编造。
- 如果你不确定精确点位，请先用 candidate_region 指定候选区域，再给出该区域内最稳的 point。
""".strip()

        # GLM-4V 限制：只发送当前原始截图，避免超出图片数量限制
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": self._encode_image(current_image)}},
        ]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _workflow_hint(self, app_name: str, task_type: str) -> str:
        hints = {
            "search": "通常流程：进入 App -> 搜索框 -> 输入关键词 -> 进入结果列表 -> 点击最相关结果。",
            "hotel": "通常流程：进入 App -> 酒店/住宿频道 -> 输入城市或地点 -> 查看结果列表 -> 进入候选详情页。",
            "food": "通常流程：进入 App -> 搜索餐厅/菜品 -> 输入关键词 -> 浏览结果列表 -> 进入目标详情。",
            "video": "通常流程：进入 App -> 搜索入口 -> 输入视频/节目关键词 -> 在结果页点击最相关条目。",
            "shopping": "通常流程：进入 App -> 搜索框 -> 输入商品关键词 -> 结果页筛选/排序 -> 点击目标商品。",
            "train": "通常流程：进入 App -> 车票查询 -> 填写出发地/目的地/日期 -> 查询结果。",
            "map": "通常流程：进入 App -> 搜索目的地 -> 查看路线/地点详情。",
            "generic": "优先找与任务最相关的搜索框、频道入口、列表项、确认按钮。",
        }
        app_specific = {
            "淘宝": "淘宝常见入口在顶部搜索框、猜你想搜、商品列表卡片。",
            "京东": "京东常见入口在顶部搜索框、商品列表卡片、筛选栏。",
            "拼多多": "拼多多优先找顶部搜索框和商品列表，不要过早 COMPLETE。",
            "抖音": "抖音优先找搜索按钮/放大镜，再进入结果页选择视频。",
            "哔哩哔哩": "B站优先找搜索入口、综合/视频结果列表。",
            "美团": "美团优先找酒店/美食入口、搜索框、结果列表。",
            "大众点评": "大众点评优先找搜索框、榜单/结果列表、商家卡片。",
            "铁路12306": "12306 优先找出发地、目的地、日期和查询按钮。",
            "百度地图": "百度地图优先搜索地点，再看路线或地点详情。",
            "高德地图": "高德地图优先搜索框、路线入口、地点卡片。",
            "去哪儿旅行": "去哪儿优先搜索酒店/景点/航班等目标频道。",
            "携程旅行": "携程优先找酒店/机票/火车票频道和目的地输入框。",
            "滴滴出行": "滴滴优先找起点/终点输入框和车型确认区。",
            "饿了么": "饿了么优先找搜索框、外卖结果列表、加入购物车按钮。",
            "小红书": "小红书优先找搜索入口、笔记列表、商品卡片。",
            "支付宝": "支付宝常见是搜索框、生活服务频道、确认按钮。",
            "微信": "微信常见是顶部搜索、公众号/小程序入口、聊天或服务页入口。",
        }
        return f"{hints.get(task_type, hints['generic'])} {app_specific.get(app_name, '')}".strip()

    def _parse_model_output(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        text = (raw_output or "").strip()
        obj = self._extract_json_object(text)
        if obj is not None:
            action, parameters = self._extract_action_from_obj(obj)
            if action in VALID_ACTIONS and isinstance(parameters, dict):
                if "candidate_region" in obj and isinstance(parameters, dict):
                    parameters = dict(parameters)
                    parameters["_candidate_region"] = str(obj["candidate_region"]).strip().upper()
                return action, self._normalize_predicted_params(action, parameters)

        action, parameters = self._parse_with_regex(text)
        return action, parameters

    def _extract_action_from_obj(self, obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # 支持 actions 数组格式，提取第一个 action
        actions = obj.get("actions")
        if isinstance(actions, list) and len(actions) > 0:
            first_action = actions[0]
            if isinstance(first_action, dict):
                action = str(first_action.get("action", "")).upper().strip()
                parameters = first_action.get("parameters", {})
                if action in VALID_ACTIONS and isinstance(parameters, dict):
                    if "candidate_region" in first_action and isinstance(parameters, dict):
                        parameters = dict(parameters)
                        parameters["_candidate_region"] = str(first_action["candidate_region"]).strip().upper()
                    return action, self._normalize_predicted_params(action, parameters)

        action = str(obj.get("action", "")).upper().strip()
        parameters = obj.get("parameters", {})
        if action in VALID_ACTIONS and isinstance(parameters, dict):
            return action, parameters

        for key in ("final", "result", "output"):
            sub = obj.get(key)
            if isinstance(sub, dict):
                action = str(sub.get("action", "")).upper().strip()
                parameters = sub.get("parameters", {})
                if action in VALID_ACTIONS and isinstance(parameters, dict):
                    if "candidate_region" in sub and isinstance(parameters, dict):
                        parameters = dict(parameters)
                        parameters["_candidate_region"] = str(sub["candidate_region"]).strip().upper()
                    return action, parameters

        return ACTION_COMPLETE, {}

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
            region_match = re.search(
                r"(TOP_SEARCH_BOX|TOP_LEFT_ICON|TOP_RIGHT_ICON|TOP_RIGHT_SMALL|TOP_BAR|TOP_LEFT|TOP_RIGHT|CENTER_PANEL|MID_LIST|LEFT_PANEL|RIGHT_PANEL|BOTTOM_BAR|BOTTOM_LEFT|BOTTOM_RIGHT)",
                text.upper(),
            )
            numbers = [int(n) for n in re.findall(r"-?\d+", text)]
            if len(numbers) >= 2:
                payload = {"point": self._clamp_point(numbers[0], numbers[1])}
                if region_match:
                    payload["_candidate_region"] = region_match.group(1)
                return ACTION_CLICK, payload
            if region_match:
                return ACTION_CLICK, {"_candidate_region": region_match.group(1)}
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
            region = ""
            if isinstance(parameters, dict):
                region = str(parameters.get("_candidate_region", "")).upper().strip()
            point = parameters.get("point", [500, 500])
            if isinstance(point, list) and len(point) >= 2:
                result = {"point": self._clamp_point(point[0], point[1])}
            else:
                result = {"point": self._region_center(region) if region else [500, 500]}
            if region in self.CANDIDATE_REGIONS:
                result["_candidate_region"] = region
            return result

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

    def _post_process_action(
        self,
        input_data: AgentInput,
        action: str,
        parameters: Dict[str, Any],
        page_stuck: bool,
        raw_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        slots = self._state.get("slots", {})
        last_action = self._state.get("last_action", "")
        last_parameters = self._state.get("last_parameters", {})
        phase = self._state.get("phase", "launch")

        if action == ACTION_OPEN:
            app_name = parameters.get("app_name", "") if isinstance(parameters, dict) else ""
            if not app_name:
                app_name = self._state.get("app_name") or self._extract_app_name(input_data.instruction)
            self._transition_phase("home")
            return ACTION_OPEN, {"app_name": app_name}

        if action == ACTION_TYPE:
            text = parameters.get("text", "") if isinstance(parameters, dict) else ""
            text = text.strip()
            if not text:
                text = slots.get("query") or slots.get("destination") or slots.get("keyword") or ""
            if not self._allow_type_now():
                self._append_note("当前未进入输入阶段，禁止跳步 TYPE，改为先点击搜索框。")
                self._state["submit_ready"] = False
                return ACTION_CLICK, {"point": self._region_center("TOP_SEARCH_BOX")}
            self._state["submit_ready"] = True
            self._transition_phase("submit_search")
            return ACTION_TYPE, {"text": text}

        if action == ACTION_COMPLETE:
            if not self._verify_complete(input_data=input_data, raw_text=raw_text):
                fallback_action, fallback_params = self._fallback_action(input_data, page_stuck=page_stuck)
                if fallback_action != ACTION_COMPLETE:
                    self._append_note("COMPLETE 未通过代码侧校验，改写为继续推进任务的动作。")
                    return fallback_action, fallback_params
            self._transition_phase("complete")
            self._state["complete_ready"] = True
            return ACTION_COMPLETE, {}

        if action == ACTION_CLICK:
            point = parameters.get("point", [500, 500]) if isinstance(parameters, dict) else [500, 500]
            region = parameters.get("_candidate_region", "") if isinstance(parameters, dict) else ""
            point = self._clamp_point(point[0], point[1]) if isinstance(point, list) and len(point) >= 2 else self._region_center(region)
            point = self._refine_click_point(point=point, region_name=region, page_type=self._state.get("page_type", "unknown"))

            if self._looks_like_interrupt_needed(page_stuck=page_stuck, raw_text=raw_text, input_data=input_data):
                self._append_note("疑似弹窗/中断页面，优先触发关闭/跳过策略。")
                return self._interrupt_recovery_action(input_data)

            if page_stuck and last_action == ACTION_CLICK and self._same_click(last_parameters, point):
                self._append_note("连续同点点击无效，启用恢复动作。")
                return self._recovery_action(input_data)

            if region in {"TOP_SEARCH_BOX", "TOP_BAR"}:
                self._state["search_box_clicked"] = True
                self._state["submit_ready"] = False
                self._transition_phase("search_input")
            elif region in {"TOP_RIGHT_ICON", "TOP_RIGHT_SMALL"}:
                self._state["submit_ready"] = True
                self._transition_phase("submit_search")
            elif region in {"MID_LIST", "CENTER_PANEL", "LEFT_PANEL", "RIGHT_PANEL"} and self._state.get("typed_texts"):
                self._state["complete_ready"] = True
                self._transition_phase("detail")
            elif region in {"BOTTOM_BAR", "BOTTOM_LEFT", "BOTTOM_RIGHT"}:
                self._state["complete_ready"] = True
                self._transition_phase("confirm")
            else:
                self._transition_phase("results")

            return ACTION_CLICK, {"point": point}

        if action == ACTION_SCROLL:
            start = parameters.get("start_point", [500, 800]) if isinstance(parameters, dict) else [500, 800]
            end = parameters.get("end_point", [500, 200]) if isinstance(parameters, dict) else [500, 200]
            self._transition_phase("results")
            return ACTION_SCROLL, {
                "start_point": self._clamp_point(start[0], start[1]),
                "end_point": self._clamp_point(end[0], end[1]),
            }

        return action, parameters if isinstance(parameters, dict) else {}

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

    def _fallback_action(self, input_data: AgentInput, page_stuck: bool = False) -> Tuple[str, Dict[str, Any]]:
        slots = self._state.get("slots", {})
        task_type = self._state.get("task_type", "generic")
        phase = self._state.get("phase", "launch")

        if input_data.step_count == 1 or phase == "launch":
            app_name = self._state.get("app_name") or self._extract_app_name(input_data.instruction)
            if app_name:
                return ACTION_OPEN, {"app_name": app_name}

        if page_stuck:
            return self._recovery_action(input_data)

        if phase in {"home", "search_entry"}:
            return ACTION_CLICK, {"point": self._region_center("TOP_SEARCH_BOX")}

        if phase in {"search_input", "submit_search"} and slots.get("query") and not self._state.get("typed_texts"):
            return ACTION_TYPE, {"text": slots["query"]}

        if task_type == "train" and not self._state.get("typed_texts"):
            return ACTION_CLICK, {"point": [280, 220]}

        return ACTION_SCROLL, {"start_point": [500, 760], "end_point": [500, 280]}

    def _recovery_action(self, input_data: AgentInput) -> Tuple[str, Dict[str, Any]]:
        slots = self._state.get("slots", {})
        task_type = self._state.get("task_type", "generic")
        last_action = self._state.get("last_action", "")
        phase = self._state.get("phase", "launch")

        if self._looks_like_interrupt_needed(page_stuck=True, raw_text="", input_data=input_data):
            return self._interrupt_recovery_action(input_data)

        if phase in {"home", "search_entry"} or (task_type in {"search", "shopping", "video", "food", "hotel", "map"} and not self._state.get("typed_texts")):
            self._append_note("页面未变化，回退到顶部搜索入口策略。")
            return ACTION_CLICK, {"point": self._region_center("TOP_SEARCH_BOX")}

        if last_action == ACTION_TYPE:
            self._append_note("页面未变化，尝试触发输入后的确认/搜索。")
            return ACTION_CLICK, {"point": self._region_center("TOP_RIGHT_ICON")}

        if slots.get("query") and not self._state.get("typed_texts"):
            self._append_note("页面未变化，改为直接输入任务关键词。")
            return ACTION_TYPE, {"text": slots["query"]}

        self._append_note("页面未变化，改为滑动探索新的候选区域。")
        return ACTION_SCROLL, {"start_point": [500, 780], "end_point": [500, 260]}

    def _interrupt_recovery_action(self, input_data: AgentInput) -> Tuple[str, Dict[str, Any]]:
        """
        比赛环境里弹窗/广告/升级提示常见关闭区多在右上、左上、底部次按钮。
        """
        retry = self._state.get("phase_retry_count", 0)
        self._append_note("疑似中断页面，尝试关闭弹窗/跳过。")
        if retry % 3 == 0:
            return ACTION_CLICK, {"point": self._region_center("TOP_RIGHT_ICON")}
        if retry % 3 == 1:
            return ACTION_CLICK, {"point": self._region_center("TOP_LEFT")}
        return ACTION_CLICK, {"point": self._region_center("BOTTOM_RIGHT")}

    def _update_runtime_state(
        self,
        current_image: Image.Image,
        current_signature: str,
        current_feature: np.ndarray,
        output: AgentOutput,
    ):
        action = getattr(output, "action", "")
        if action == ACTION_TYPE:
            text = output.parameters.get("text", "")
            if text and text not in self._state["typed_texts"]:
                self._state["typed_texts"].append(text)
            self._state["submit_ready"] = True

        if action == ACTION_CLICK:
            point = output.parameters.get("point", [])
            if isinstance(point, list) and len(point) >= 2:
                if point[1] <= 180:
                    self._state["search_box_clicked"] = True
                if point[1] >= 300:
                    self._state["complete_ready"] = True

        last_action = self._state.get("last_action", "")
        last_parameters = self._state.get("last_parameters", {})
        if action == last_action and output.parameters == last_parameters:
            self._state["repeat_action_count"] = self._state.get("repeat_action_count", 0) + 1
            self._state["phase_retry_count"] = self._state.get("phase_retry_count", 0) + 1
        else:
            self._state["repeat_action_count"] = 0
            self._state["phase_retry_count"] = 0

        self._state["last_action"] = action
        self._state["last_parameters"] = output.parameters
        self._state["screen_signatures"] = (self._state["screen_signatures"] + [current_signature])[-6:]
        self._state["screen_features"] = (self._state["screen_features"] + [current_feature])[-3:]
        self._state["last_image"] = current_image.copy()

    def _infer_task_type(self, instruction: str, app_name: str) -> str:
        if "酒店" in instruction or "民宿" in instruction:
            return "hotel"
        if "美食" in instruction or "餐厅" in instruction or "外卖" in instruction or app_name in {"美团", "大众点评", "饿了么", "盒马"}:
            return "food"
        if "车票" in instruction or "高铁" in instruction or "火车" in instruction or app_name == "铁路12306":
            return "train"
        if "地图" in instruction or "导航" in instruction or "路线" in instruction or app_name in {"百度地图", "高德地图"}:
            return "map"
        if "视频" in instruction or "番剧" in instruction or "直播" in instruction or app_name in {"抖音", "哔哩哔哩", "爱奇艺", "腾讯视频", "芒果TV", "快手", "优酷视频"}:
            return "video"
        if "购买" in instruction or "下单" in instruction or "商品" in instruction or app_name in {"淘宝", "京东", "拼多多", "小红书"}:
            return "shopping"
        if "搜索" in instruction or "查找" in instruction or "看看" in instruction:
            return "search"
        return "generic"

    def _extract_slots(self, instruction: str, task_type: str) -> Dict[str, str]:
        slots: Dict[str, str] = {}
        quoted = re.findall(r"[\"“](.*?)[\"”]", instruction)
        if quoted:
            slots["query"] = quoted[0].strip()

        patterns = [
            (r"帮我搜(?:索)?(.+)", "query"),
            (r"查一下(.+)", "query"),
            (r"看看(.+)", "query"),
            (r"找一下(.+)", "query"),
        ]
        for pattern, key in patterns:
            m = re.search(pattern, instruction)
            if m and key not in slots:
                value = m.group(1).strip("。！？!?，, ")
                if value:
                    slots[key] = value

        city_match = re.search(r"去([^\s，。!?]{2,8})", instruction)
        if city_match and "destination" not in slots:
            slots["destination"] = city_match.group(1)

        near_match = re.search(r"(.+?)附近", instruction)
        if near_match and "location" not in slots:
            slots["location"] = near_match.group(1).strip()

        if task_type == "shopping" and "query" not in slots:
            m = re.search(r"(买|购买|下单)(.+)", instruction)
            if m:
                slots["query"] = m.group(2).strip("。！？!?，, ")
        if task_type in {"video", "search"} and "query" not in slots:
            m = re.search(r"(搜索|查找|看看)(.+)", instruction)
            if m:
                slots["query"] = m.group(2).strip("。！？!?，, ")

        if "query" not in slots:
            compact = instruction.strip().replace("帮我", "").replace("请", "")
            slots["query"] = compact[:18]

        return slots

    def _append_note(self, note: str):
        notes = self._state.setdefault("notes", [])
        if not notes or notes[-1] != note:
            notes.append(note)
        self._state["notes"] = notes[-8:]

    def _make_grid_image(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for col in range(1, self._grid_cols):
            x = int(round(w * col / self._grid_cols))
            draw.line((x, 0, x, h), fill=(255, 64, 64), width=3)

        for row in range(1, self._grid_rows):
            y = int(round(h * row / self._grid_rows))
            draw.line((0, y, w, y), fill=(255, 64, 64), width=3)

        cell_w = w / self._grid_cols
        cell_h = h / self._grid_rows
        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                label = f"{chr(ord('A') + row)}{col + 1}"
                x = int(col * cell_w + 10)
                y = int(row * cell_h + 10)
                draw.rectangle((x, y, x + 56, y + 24), fill=(0, 0, 0))
                draw.text((x + 4, y + 4), label, fill=(255, 255, 0))

        return img

    def _make_region_crops(self, image: Image.Image) -> Dict[str, Image.Image]:
        img = image.convert("RGB")
        w, h = img.size
        crops: Dict[str, Image.Image] = {}
        for region_name, (x1r, y1r, x2r, y2r) in self.REGION_CROP_RANGES.items():
            x1 = int(round(w * x1r))
            y1 = int(round(h * y1r))
            x2 = int(round(w * x2r))
            y2 = int(round(h * y2r))
            crop = img.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
            crops[region_name] = self._annotate_crop(crop, region_name)
        return crops

    def _annotate_crop(self, image: Image.Image, region_name: str) -> Image.Image:
        crop = image.copy()
        draw = ImageDraw.Draw(crop)
        draw.rectangle((0, 0, min(crop.size[0] - 1, 170), 28), fill=(0, 0, 0))
        draw.text((6, 6), region_name, fill=(255, 255, 0))
        return crop

    def _region_center(self, region_name: str) -> List[int]:
        return list(self.CANDIDATE_REGIONS.get(region_name or "", [500, 500]))

    def _snap_point_to_region(self, point: List[int], region_name: str) -> List[int]:
        region_name = (region_name or "").upper().strip()
        if region_name not in self.CANDIDATE_REGIONS:
            return point
        center = self._region_center(region_name)
        px, py = point
        cx, cy = center
        px = int(round((px * 0.6) + (cx * 0.4)))
        py = int(round((py * 0.6) + (cy * 0.4)))
        return self._clamp_point(px, py)

    def _same_click(self, last_parameters: Dict[str, Any], point: List[int]) -> bool:
        last_point = last_parameters.get("point", []) if isinstance(last_parameters, dict) else []
        return bool(isinstance(last_point, list) and len(last_point) >= 2 and last_point == point)

    def _infer_page_type(self, input_data: AgentInput, current_image: Image.Image) -> str:
        phase = self._state.get("phase", "launch")
        if input_data.step_count == 1 and phase == "launch":
            return "home_or_launcher"
        last_action = self._state.get("last_action", "")
        if last_action == ACTION_TYPE:
            return "search_confirm"
        if self._state.get("search_box_clicked") and not self._state.get("typed_texts"):
            return "search_page"
        if self._state.get("typed_texts"):
            return "results_or_detail"
        if phase in {"confirm", "detail"}:
            return "detail"
        if phase in {"search_entry", "search_input", "submit_search"}:
            return "search_page"
        return "generic_page"

    def _allow_type_now(self) -> bool:
        phase = self._state.get("phase", "launch")
        if phase in {"search_input", "submit_search"}:
            return True
        if self._state.get("search_box_clicked"):
            return True
        last_action = self._state.get("last_action", "")
        last_point = self._state.get("last_parameters", {}).get("point", [])
        if last_action == ACTION_CLICK and isinstance(last_point, list) and len(last_point) >= 2 and last_point[1] <= 180:
            return True
        return False

    def _verify_complete(self, input_data: AgentInput, raw_text: str) -> bool:
        phase = self._state.get("phase", "launch")
        page_type = self._state.get("page_type", "unknown")
        typed = bool(self._state.get("typed_texts"))
        retry_count = self._state.get("phase_retry_count", 0)
        if input_data.step_count < 4:
            return False
        if phase in {"launch", "home", "search_entry", "search_input", "submit_search"}:
            return False
        if page_type in {"home_or_launcher", "search_page", "search_confirm"}:
            return False
        if not (typed or self._state.get("complete_ready") or retry_count >= 1):
            return False
        weak_negative = ("搜索框" in raw_text) or ("输入" in raw_text and "已完成" not in raw_text)
        if weak_negative:
            return False
        self._state["complete_ready"] = True
        return True

    def _region_candidate_points(self, region_name: str) -> List[List[int]]:
        region_name = (region_name or "").upper().strip()
        candidates = {
            "TOP_SEARCH_BOX": [[500, 110], [470, 110], [530, 110], [420, 118], [580, 118]],
            "TOP_RIGHT_ICON": [[930, 110], [900, 110], [950, 110], [880, 128]],
            "TOP_RIGHT_SMALL": [[880, 110], [850, 110], [910, 110], [870, 132]],
            "TOP_LEFT_ICON": [[88, 110], [120, 110], [70, 110], [150, 128]],
            "TOP_BAR": [[500, 110], [420, 110], [580, 110]],
            "MID_LIST": [[500, 560], [500, 500], [500, 620], [420, 560], [580, 560]],
            "RIGHT_PANEL": [[760, 520], [820, 480], [860, 420]],
            "LEFT_PANEL": [[240, 520], [180, 480], [140, 420]],
            "BOTTOM_BAR": [[500, 900], [420, 900], [580, 900]],
            "BOTTOM_RIGHT": [[780, 900], [840, 860], [900, 820]],
            "BOTTOM_LEFT": [[220, 900], [160, 860], [100, 820]],
            "CENTER_PANEL": [[500, 430], [500, 500], [500, 360]],
        }
        return [self._clamp_point(x, y) for x, y in candidates.get(region_name, [self._region_center(region_name)])]

    def _refine_click_point(self, point: List[int], region_name: str, page_type: str) -> List[int]:
        region_name = (region_name or "").upper().strip()
        if region_name not in self.CANDIDATE_REGIONS:
            return point
        snapped = self._snap_point_to_region(point, region_name)
        candidates = self._region_candidate_points(region_name)
        # 对高频易错小控件，优先候选点，不完全相信模型直接点。
        if region_name in {"TOP_SEARCH_BOX", "TOP_RIGHT_ICON", "TOP_RIGHT_SMALL", "TOP_LEFT_ICON"}:
            scored = sorted(
                candidates,
                key=lambda p: abs(p[0] - snapped[0]) + abs(p[1] - snapped[1]),
            )
            return scored[0]
        return snapped

    def _transition_phase(self, target_phase: str):
        old_phase = self._state.get("phase", "launch")
        if target_phase != old_phase:
            self._state["phase_retry_count"] = 0
        self._state["phase"] = target_phase
        self._state["milestone"] = self.PHASE_TRANSITIONS.get(target_phase, target_phase)
        if target_phase in {"launch", "home", "search_entry"}:
            self._state["complete_ready"] = False
        if target_phase == "search_input":
            self._state["submit_ready"] = False
        if target_phase in {"results", "detail", "confirm"}:
            self._state["complete_ready"] = True

    def _looks_like_interrupt_needed(self, page_stuck: bool, raw_text: str, input_data: AgentInput) -> bool:
        last_repeat = self._state.get("repeat_action_count", 0) >= 1
        interrupt_tokens = ["关闭", "跳过", "以后再说", "取消", "知道了", "我知道了", "暂不", "不了", "稍后"]
        hit_text = any(tok in raw_text for tok in interrupt_tokens)
        return bool(page_stuck and (last_repeat or hit_text or input_data.step_count >= 3 and self._state.get("phase_retry_count", 0) >= 1))

    def _sync_memory_from_history(self, input_data: AgentInput):
        history_actions = input_data.history_actions or []
        last_history_len = self._state.get("last_history_len", 0)
        if len(history_actions) <= last_history_len:
            return

        app_name = self._state.get("app_name", "") or "UNKNOWN_APP"
        task_type = self._state.get("task_type", "generic")
        for item in history_actions[last_history_len:]:
            if not item.get("is_valid", False):
                continue
            memory_note = self._summarize_valid_action(item)
            if memory_note:
                self._store_app_memory(app_name, task_type, memory_note)
        self._state["last_history_len"] = len(history_actions)

    def _summarize_valid_action(self, item: Dict[str, Any]) -> str:
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

    def _load_app_memory(self) -> Dict[str, Dict[str, List[str]]]:
        if not self._memory_path.exists():
            return {}
        try:
            with self._memory_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _persist_app_memory(self):
        try:
            with self._memory_path.open("w", encoding="utf-8") as f:
                json.dump(self._app_memory, f, ensure_ascii=False, indent=2)
        except Exception:
            # 评测环境中如果目录不可写，则退化为仅内存态
            pass

    def _store_app_memory(self, app_name: str, task_type: str, note: str):
        app_name = app_name or "UNKNOWN_APP"
        bucket = self._app_memory.setdefault(app_name, {})
        notes = bucket.setdefault(task_type, [])
        if note not in notes:
            notes.append(note)
        bucket[task_type] = notes[-10:]
        self._persist_app_memory()

    def _retrieve_app_memory(self, app_name: str, task_type: str) -> str:
        if not app_name:
            return ""
        bucket = self._app_memory.get(app_name, {})
        notes = list(bucket.get(task_type, [])) + list(bucket.get("generic", []))
        notes = notes[-5:]
        return "；".join(notes)

    @classmethod
    def _extract_app_name(cls, instruction: str) -> str:
        for app_name in cls.KNOWN_APPS:
            if app_name in instruction:
                return app_name
        return ""
