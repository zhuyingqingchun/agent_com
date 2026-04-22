"""
任务完成判断模块 - 基于关键词提取的智能完成检测

核心思路：
1. 从指令中提取"完成目标"（店铺名、商品名、目的地等）
2. 在状态机中追踪目标达成进度
3. 结合页面特征和历史动作判断是否可以完成
"""

import re
from typing import Any, Dict, List, Optional


class CompletionTracker:
    """任务完成追踪器"""
    
    def __init__(self, state: Dict[str, Any]):
        self.state = state
        self.slots = state.get("slots", {})
        self.task_type = state.get("task_type", "generic")
        self.instruction = state.get("instruction", "")
    
    def get_completion_targets(self) -> Dict[str, Any]:
        """从已提取的关键词中获取完成目标"""
        targets = {
            "app_name": self.state.get("app_name", ""),
            "shop_name": self.slots.get("shop_name", ""),
            "product_name": self.slots.get("product_name", ""),
            "search_query": self.slots.get("query", ""),
            "task_type": self.task_type,
        }
        
        # 根据任务类型补充目标
        if self.task_type == "map":
            targets["destination"] = self._extract_destination()
        elif self.task_type == "hotel":
            targets["hotel_name"] = self.slots.get("query", "")
        elif self.task_type == "train":
            targets["route"] = self._extract_route()
        
        return targets
    
    def _extract_destination(self) -> str:
        """提取目的地（地图类任务）"""
        # 从指令中提取目的地
        patterns = [
            r"去(.+?)(?:的?路线|导航|怎么走)",
            r"导航到(.+?)(?:，|,|。|！|？|$)",
            r"从.+?到(.+?)(?:，|,|。|！|？|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, self.instruction)
            if match:
                return match.group(1).strip()
        return self.slots.get("query", "")
    
    def _extract_route(self) -> Dict[str, str]:
        """提取路线信息（火车票类任务）"""
        route = {"from": "", "to": ""}
        # 从XX到XX
        match = re.search(r"从(.+?)[到至](.+?)(?:的?票|，|,|。|！|？|$)", self.instruction)
        if match:
            route["from"] = match.group(1).strip()
            route["to"] = match.group(2).strip()
        return route
    
    def calculate_completion_score(self, raw_text: str, history: List[Dict[str, Any]]) -> float:
        """计算任务完成度分数 [0.0, 1.0]"""
        targets = self.get_completion_targets()
        score = 0.0
        checks = []
        
        # 基础检查：应用已打开
        if self.state.get("phase") not in ["launch", "home"]:
            score += 0.2
            checks.append("app_opened")
        
        # 搜索已执行
        if self.state.get("typed_texts"):
            score += 0.2
            checks.append("searched")
        
        # 根据任务类型检查特定目标
        if self.task_type == "food":
            score += self._check_food_completion(targets, raw_text, checks)
        elif self.task_type == "shopping":
            score += self._check_shopping_completion(targets, raw_text, checks)
        elif self.task_type == "map":
            score += self._check_map_completion(targets, raw_text, checks)
        elif self.task_type == "hotel":
            score += self._check_hotel_completion(targets, raw_text, checks)
        elif self.task_type == "video":
            score += self._check_video_completion(targets, raw_text, checks)
        else:
            # 通用任务：检查搜索结果
            score += self._check_generic_completion(targets, raw_text, checks)
        
        # 历史动作检查
        if len(history) >= 3:
            score += 0.1
            checks.append("has_history")
        
        # 页面特征检查
        if self._is_result_page(raw_text):
            score += 0.1
            checks.append("result_page")
        
        return min(score, 1.0), checks
    
    def _check_food_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查美食任务完成度"""
        score = 0.0
        shop_name = targets.get("shop_name", "")
        product_name = targets.get("product_name", "")
        
        # 检查店铺名是否出现
        if shop_name and self._is_target_in_text(shop_name, raw_text):
            score += 0.25
            checks.append("shop_found")
        
        # 检查商品名是否出现
        if product_name and self._is_target_in_text(product_name, raw_text):
            score += 0.25
            checks.append("product_found")
        
        # 如果只提取到搜索词，检查搜索词
        if not shop_name and not product_name:
            query = targets.get("search_query", "")
            if query and self._is_target_in_text(query, raw_text):
                score += 0.3
                checks.append("query_found")
        
        return score
    
    def _check_shopping_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查购物任务完成度"""
        score = 0.0
        product_name = targets.get("product_name", "")
        shop_name = targets.get("shop_name", "")
        
        if product_name and self._is_target_in_text(product_name, raw_text):
            score += 0.3
            checks.append("product_found")
        
        if shop_name and self._is_target_in_text(shop_name, raw_text):
            score += 0.2
            checks.append("shop_found")
        
        return score
    
    def _check_map_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查地图任务完成度"""
        score = 0.0
        destination = targets.get("destination", "")
        
        if destination and self._is_target_in_text(destination, raw_text):
            score += 0.4
            checks.append("destination_found")
        
        # 检查是否有路线信息
        if "路线" in raw_text or "距离" in raw_text or "公里" in raw_text:
            score += 0.1
            checks.append("route_info")
        
        return score
    
    def _check_hotel_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查酒店任务完成度"""
        score = 0.0
        hotel_name = targets.get("hotel_name", "")
        
        if hotel_name and self._is_target_in_text(hotel_name, raw_text):
            score += 0.4
            checks.append("hotel_found")
        
        # 检查酒店特征
        if any(kw in raw_text for kw in ["入住", "离店", "价格", "预订"]):
            score += 0.1
            checks.append("hotel_page")
        
        return score
    
    def _check_video_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查视频任务完成度"""
        score = 0.0
        query = targets.get("search_query", "")
        
        if query and self._is_target_in_text(query, raw_text):
            score += 0.3
            checks.append("video_found")
        
        # 检查视频特征
        if any(kw in raw_text for kw in ["播放", "弹幕", "点赞", "收藏"]):
            score += 0.2
            checks.append("video_page")
        
        return score
    
    def _check_generic_completion(self, targets: Dict[str, Any], raw_text: str, checks: List[str]) -> float:
        """检查通用任务完成度"""
        score = 0.0
        query = targets.get("search_query", "")
        
        if query and self._is_target_in_text(query, raw_text):
            score += 0.4
            checks.append("target_found")
        
        return score
    
    def _is_target_in_text(self, target: str, text: str) -> bool:
        """检查目标是否出现在文本中（支持模糊匹配）"""
        if not target or not text:
            return False
        
        target = target.lower()
        text = text.lower()
        
        # 完全匹配
        if target in text:
            return True
        
        # 部分匹配（目标长度的80%以上）
        if len(target) >= 4:
            for i in range(len(target) - 3):
                if target[i:i+4] in text:
                    return True
        
        return False
    
    def _is_result_page(self, raw_text: str) -> bool:
        """判断是否为结果页面"""
        result_indicators = [
            "结果", "列表", "详情", "页面", "找到", "相关",
            "price", "评分", "地址", "电话", "营业时间"
        ]
        return any(indicator in raw_text.lower() for indicator in result_indicators)
    
    def should_complete(self, input_data: Any, raw_text: str, min_score: float = 0.7) -> tuple:
        """判断是否应该完成任务
        
        Returns:
            (should_complete: bool, score: float, reasons: List[str])
        """
        step_count = getattr(input_data, 'step_count', 0)
        history = getattr(input_data, 'history_actions', [])
        phase = self.state.get("phase", "launch")
        
        # 基础条件检查
        if step_count < 3:
            return False, 0.0, ["too_early"]
        
        if phase in ["launch", "home"]:
            return False, 0.0, ["not_started"]
        
        # 计算完成度
        score, checks = self.calculate_completion_score(raw_text, history)
        
        # 检查负面指标
        negative_indicators = [
            ("搜索框" in raw_text, "still_at_search"),
            ("输入" in raw_text and "已完成" not in raw_text, "need_input"),
            ("登录" in raw_text or "注册" in raw_text, "need_login"),
            ("错误" in raw_text or "失败" in raw_text, "has_error"),
        ]
        
        negatives = [reason for condition, reason in negative_indicators if condition]
        
        # 如果分数足够高且没有负面指标，允许完成
        if score >= min_score and not negatives:
            return True, score, checks

        return False, score, checks + negatives


def verify_complete_v2(state: Dict[str, Any], input_data: Any, raw_text: str) -> bool:
    """新版完成验证函数（基于关键词提取）"""
    tracker = CompletionTracker(state)
    should_complete, score, reasons = tracker.should_complete(input_data, raw_text)
    
    if state.get("verbose_logging"):
        print(f"[完成判断] 分数={score:.2f}, 原因={reasons}, 结果={should_complete}")
    
    if should_complete:
        state["complete_ready"] = True
        state["completion_score"] = score
        state["completion_reasons"] = reasons
    
    return should_complete


def get_completion_hint(state: Dict[str, Any]) -> str:
    """获取完成提示（用于prompt）"""
    tracker = CompletionTracker(state)
    targets = tracker.get_completion_targets()
    
    hints = []
    
    if targets.get("shop_name"):
        hints.append(f"目标店铺：{targets['shop_name']}")
    if targets.get("product_name"):
        hints.append(f"目标商品：{targets['product_name']}")
    if targets.get("destination"):
        hints.append(f"目标地点：{targets['destination']}")
    if targets.get("search_query"):
        hints.append(f"搜索关键词：{targets['search_query']}")
    
    if hints:
        return "【完成目标】" + "；".join(hints)
    return ""
