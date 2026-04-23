"""
点击坐标专用 Prompt 模块

核心原则：
1. 所有坐标统一使用归一化坐标 [0, 0] 到 [1000, 1000]
2. 根据截图中UI元素的实际位置动态判断坐标
3. 不提供固定UI元素位置，避免误导
"""

from typing import Any, Dict, List, Optional


class ClickPromptHelper:
    """点击坐标 Prompt 辅助类"""
    
    # 坐标系说明 - 统一归一化坐标
    COORDINATE_SYSTEM = """
【坐标系统 - 重要】
- 坐标范围: [0, 0] 到 [1000, 1000]（统一归一化坐标）
- 原点 [0, 0]: 屏幕左上角
- 中心点 [500, 500]: 屏幕正中央
- X轴: 从左到右递增（0=最左, 1000=最右）
- Y轴: 从上到下递增（0=最上, 1000=最下）

【坐标输出规则】
1. 必须输出归一化坐标 [x, y]，范围 0-1000
2. 根据截图中UI元素的实际位置判断坐标
3. 点击UI元素的中心位置
4. 不要假设固定位置，每个截图的UI布局可能不同
"""

    DIRECTION_HINTS = {
        "home": "目标通常在屏幕顶部（搜索栏/入口图标）或底部操作栏",
        "search_entry": "目标通常是屏幕顶部的搜索框或搜索图标",
        "search_input": "目标是当前已激活的输入框本体，在顶部区域",
        "submit_search": "目标通常是右上角的执行按钮（搜索/确认），或在输入框右侧",
        "results": "目标在屏幕中部的结果列表里",
        "detail": "如果任务还需要搜索具体商品/菜品，优先看顶部是否有店内搜索框；如果需要选择规格/数量/确认下单，目标通常在中部内容区或底部",
        "confirm": "目标通常是底部的确认/发送/提交按钮",
    }

    @classmethod
    def get_click_guidance(cls, target_description: str = "", phase: str = "") -> str:
        guidance_parts = [cls.COORDINATE_SYSTEM]
        guidance_parts.append("\n【点击规则】")
        guidance_parts.append("1. 只找当前最该点击的那个控件")
        guidance_parts.append("2. 点击控件中心，不要贴边")
        guidance_parts.append("3. 如果是顶部小图标，不要点到最上沿空白")
        
        if target_description:
            guidance_parts.append(f"\n【当前点击目标】")
            guidance_parts.append(f"目标: {target_description}")
            guidance_parts.append("在截图中找到该元素，点击其中心位置")

        direction_hint = cls.DIRECTION_HINTS.get(phase, "")
        if direction_hint:
            guidance_parts.append(f"\n【方向参考（仅作方位提示，坐标以截图实际位置为准）】")
            guidance_parts.append(direction_hint)
            guidance_parts.append("以上只是大致方位提示，你必须根据截图中的实际UI元素位置来确定精确坐标。")

        return "\n".join(guidance_parts)
    
    @classmethod
    def get_click_prompt_for_action(
        cls,
        instruction: str,
        state: Dict[str, Any],
        target_element: str = "",
    ) -> str:
        """为 CLICK 动作生成完整的 Prompt
        
        Args:
            instruction: 任务指令
            state: 当前状态
            target_element: 目标元素描述
        
        Returns:
            完整的点击 Prompt
        """
        phase = state.get("phase", "unknown")
        
        # 根据阶段推断点击目标
        if not target_element:
            target_element = cls._infer_target_from_phase(phase, state)
        
        # 构建 Prompt
        prompt_parts = [
            f"【任务】{instruction}",
            f"【动作】CLICK - 点击屏幕上的元素",
            cls.get_click_guidance(target_element),
            "\n【输出要求】",
            "只输出JSON，不要任何解释:",
            "优先点击候选区域中心，不要贴区域边缘:",
            '{"action":"CLICK","parameters":{"point":[x,y]}}',
        ]
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def _infer_target_from_phase(cls, phase: str, state: Dict[str, Any]) -> str:
        last_action = state.get("last_action", "")
        typed_texts = state.get("typed_texts", [])
        last_params = state.get("last_parameters", {})

        if last_action == "TYPE":
            typed = last_params.get("text", "")
            preview = typed[:12] + ".." if len(typed) > 12 else typed
            return f"确认/提交/发送按钮（用于提交「{preview}」的输入内容）"
        if last_action == "CLICK":
            last_point = last_params.get("point", [])
            is_top_click = (
                isinstance(last_point, list) and len(last_point) >= 2 
                and last_point[1] <= 220
            )
            if is_top_click:
                return "已激活的输入框本体（可直接TYPE，不需要再点击激活）"
            if typed_texts:
                last_typed = typed_texts[-1] if typed_texts else ""
                preview = last_typed[:10] + ".." if len(last_typed) > 10 else last_typed
                return f"目标操作项或下一步控件（已输入过「{preview}」，可能需要点结果项、确认或继续操作）"
        
        if typed_texts:
            last_typed = typed_texts[-1] if typed_texts else ""
            preview = last_typed[:10] + ".." if len(last_typed) > 10 else last_typed
            return f"目标元素或确认控件（已有输入记录「{preview}」）"
        
        phase_targets = {
            "launch": "应用图标",
            "home": "任务主入口或可编辑入口",
            "search_entry": "可编辑区域（输入框/文本框）",
            "results": "目标结果项或下一步功能入口",
            "confirm": "最终执行按钮（发送/提交/确认/发布，通常在底部右侧）",
        }
        return phase_targets.get(phase, "相关UI元素")


def enhance_action_prompt_with_click_guidance(
    original_prompt: str,
    action: str,
    state: Dict[str, Any],
    instruction: str,
) -> str:
    """增强 Action Prompt，为 CLICK 动作添加坐标指导
    
    Args:
        original_prompt: 原始 Prompt
        action: 预测的动作类型
        state: 当前状态
        instruction: 任务指令
    
    Returns:
        增强后的 Prompt
    """
    if action != "CLICK":
        return original_prompt
    
    # 为 CLICK 动作添加坐标指导
    click_guidance = ClickPromptHelper.get_click_prompt_for_action(
        instruction=instruction,
        state=state,
    )
    
    return click_guidance
