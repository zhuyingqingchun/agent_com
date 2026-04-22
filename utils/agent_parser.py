"""
模型输出解析模块 - 支持多种解析策略
"""
import json
import re
from typing import Any, Dict, Optional, Tuple

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE, VALID_ACTIONS
from utils.agent_regions import clamp_point, region_center


class OutputParser:
    """模型输出解析器"""
    
    def __init__(self, candidate_regions: Dict[str, list]):
        self.candidate_regions = candidate_regions
    
    def parse(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """解析模型输出，返回 (action, parameters)"""
        text = (raw_output or "").strip()
        obj = self._extract_json_object(text)
        if obj is not None:
            action, parameters = self._extract_action_from_obj(obj)
            if action in VALID_ACTIONS and isinstance(parameters, dict):
                if "candidate_region" in obj:
                    parameters = dict(parameters)
                    parameters["_candidate_region"] = str(obj["candidate_region"]).strip().upper()
                return action, self._normalize_params(action, parameters)
        
        # 备用：正则解析
        return self._parse_with_regex(text)
    
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取 JSON 对象或数组"""
        candidates = [text]
        # 尝试匹配 JSON 对象
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            candidates.append(match.group(0))
        # 尝试匹配 JSON 数组
        array_match = re.search(r"\[.*\]", text, flags=re.S)
        if array_match:
            candidates.append(array_match.group(0))
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
                # 如果是数组，取第一个元素
                if isinstance(obj, list) and obj:
                    first = obj[0]
                    if isinstance(first, dict):
                        return first
            except Exception:
                continue
        return None
    
    def _extract_action_from_obj(self, obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """从 JSON 对象提取动作"""
        # 尝试 actions 数组格式
        actions = obj.get("actions")
        if isinstance(actions, list) and actions:
            first_action = actions[0]
            if isinstance(first_action, dict):
                action = str(first_action.get("action", "")).upper().strip()
                parameters = first_action.get("parameters", {})
                if action in VALID_ACTIONS and isinstance(parameters, dict):
                    if "candidate_region" in first_action:
                        parameters = dict(parameters)
                        parameters["_candidate_region"] = str(first_action["candidate_region"]).strip().upper()
                    return action, self._normalize_params(action, parameters)
        
        # 直接提取 action 和 parameters
        action = str(obj.get("action", "")).upper().strip()
        parameters = obj.get("parameters", {})
        if action in VALID_ACTIONS and isinstance(parameters, dict):
            return action, self._normalize_params(action, parameters)
        
        # 尝试嵌套字段
        for key in ("final", "result", "output"):
            sub = obj.get(key)
            if isinstance(sub, dict):
                action = str(sub.get("action", "")).upper().strip()
                parameters = sub.get("parameters", {})
                if action in VALID_ACTIONS and isinstance(parameters, dict):
                    if "candidate_region" in sub:
                        parameters = dict(parameters)
                        parameters["_candidate_region"] = str(sub["candidate_region"]).strip().upper()
                    return action, self._normalize_params(action, parameters)
        
        return ACTION_COMPLETE, {}
    
    def _parse_with_regex(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """使用正则表达式解析（备用方案）"""
        action_match = re.search(r"(CLICK|TYPE|SCROLL|OPEN|COMPLETE)", text.upper())
        action = action_match.group(1) if action_match else ACTION_COMPLETE
        
        if action == ACTION_CLICK:
            region_match = re.search(
                r"(TOP_SEARCH_BOX|TOP_LEFT_ICON|TOP_RIGHT_ICON|TOP_RIGHT_SMALL|TOP_BAR|TOP_LEFT|TOP_RIGHT|CENTER_PANEL|MID_LIST|LEFT_PANEL|RIGHT_PANEL|BOTTOM_BAR|BOTTOM_LEFT|BOTTOM_RIGHT)",
                text.upper(),
            )
            numbers = [int(n) for n in re.findall(r"-?\d+", text)]
            if len(numbers) >= 2:
                payload = {"point": clamp_point(numbers[0], numbers[1])}
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
                    "start_point": clamp_point(numbers[0], numbers[1]),
                    "end_point": clamp_point(numbers[2], numbers[3]),
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
    
    def _normalize_params(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """规范化预测参数"""
        if action == ACTION_CLICK:
            region = str(parameters.get("_candidate_region", "")).upper().strip() if isinstance(parameters, dict) else ""
            point = parameters.get("point", [500, 500])
            if isinstance(point, list) and len(point) >= 2:
                result = {"point": clamp_point(point[0], point[1])}
            else:
                result = {"point": region_center(region, self.candidate_regions) if region else [500, 500]}
            if region in self.candidate_regions:
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
                "start_point": clamp_point(start[0], start[1]),
                "end_point": clamp_point(end[0], end[1]),
            }
        
        return parameters if isinstance(parameters, dict) else {}
