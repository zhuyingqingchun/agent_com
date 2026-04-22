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

    REGION_HINTS = {
        "TOP_LEFT_ICON": ("左上角小图标/返回/关闭", "x≈40-220, y≈35-90"),
        "TOP_SEARCH_BOX": ("顶部横向搜索框/输入框", "x≈190-840, y≈50-110"),
        "TOP_RIGHT_ICON": ("右上角搜索/放大镜/更多", "x≈820-950, y≈35-85"),
        "TOP_RIGHT_SMALL": ("右上角更小的按钮/图标", "x≈800-920, y≈35-85"),
        "LEFT_PANEL": ("左中区域/左侧列表/左侧筛选", "x≈80-380, y≈260-700"),
        "CENTER_PANEL": ("中部主内容区域", "x≈300-700, y≈220-760"),
        "RIGHT_PANEL": ("右中区域/结果列表右侧", "x≈620-940, y≈260-700"),
        "MID_LIST": ("中部列表项", "x≈120-880, y≈260-760"),
        "BOTTOM_BAR": ("底部操作栏", "x≈160-840, y≈820-960"),
        "BOTTOM_RIGHT": ("底部右侧搜索/发送/确认按钮", "x≈760-980, y≈820-960"),
        "BOTTOM_LEFT": ("底部左侧入口/返回频道", "x≈20-320, y≈820-960"),
    }

    PHASE_REGION_PRIORS = {
        "home": ["TOP_RIGHT_ICON", "TOP_SEARCH_BOX", "BOTTOM_RIGHT"],
        "search_entry": ["TOP_SEARCH_BOX", "MID_LIST"],
        "search_input": ["TOP_SEARCH_BOX"],
        "submit_search": ["TOP_RIGHT_ICON", "BOTTOM_RIGHT"],
        "results": ["MID_LIST", "CENTER_PANEL", "RIGHT_PANEL", "LEFT_PANEL"],
        "detail": ["CENTER_PANEL", "BOTTOM_RIGHT", "MID_LIST"],
        "confirm": ["BOTTOM_RIGHT", "BOTTOM_BAR", "CENTER_PANEL"],
    }
    
    @classmethod
    def get_click_guidance(cls, target_description: str = "", phase: str = "") -> str:
        """获取点击坐标指导
        
        Args:
            target_description: 点击目标的描述
        
        Returns:
            点击坐标指导文本
        """
        guidance_parts = [cls.COORDINATE_SYSTEM]
        guidance_parts.append("\n【点击规则】")
        guidance_parts.append("1. 只找当前最该点击的那个控件")
        guidance_parts.append("2. 点击控件中心，不要贴边")
        guidance_parts.append("3. 如果是顶部小图标，不要点到最上沿空白")
        
        if target_description:
            guidance_parts.append(f"\n【当前点击目标】")
            guidance_parts.append(f"目标: {target_description}")
            guidance_parts.append("在截图中找到该元素，点击其中心位置")
            if "搜索入口" in target_description or "搜索按钮" in target_description:
                guidance_parts.append("右上角搜索图标通常在右上区域中间偏下")
            if "搜索框" in target_description or "输入" in target_description:
                guidance_parts.append("搜索框通常是顶部横向长条，常见区域 x≈190-840, y≈50-110")
            if "确认" in target_description or "提交" in target_description:
                guidance_parts.append("提交/确认控件常比顶部搜索图标更靠下")

        region_keys = cls.PHASE_REGION_PRIORS.get(phase, [])
        if region_keys:
            guidance_parts.append("\n【本阶段优先参考区域】")
            for key in region_keys:
                label, coord = cls.REGION_HINTS[key]
                guidance_parts.append(f"- {key}: {label}; {coord}")
            guidance_parts.append("优先在这些区域里找目标；如果截图明显不符合，再按实际位置判断。")

        if phase == "submit_search":
            guidance_parts.append("\n【提交搜索阶段补充】")
            guidance_parts.append("- 输入完成后，优先找更靠下的搜索/确认/执行控件")
            guidance_parts.append("- 常见区域1: x≈750-950, y≈105-150")
            guidance_parts.append("- 常见区域2: x≈20-980, y≈160-250")

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
            f"【当前阶段】{phase}",
            f"【动作】CLICK - 点击屏幕上的元素",
            cls.get_click_guidance(target_element, phase=phase),
            "\n【输出要求】",
            "只输出JSON，不要任何解释:",
            "优先点击候选区域中心，不要贴区域边缘:",
            '{"action":"CLICK","parameters":{"point":[x,y]}}',
        ]
        
        return "\n".join(prompt_parts)
    
    @classmethod
    def _infer_target_from_phase(cls, phase: str, state: Dict[str, Any]) -> str:
        """根据阶段推断点击目标"""
        phase_targets = {
            "launch": "应用图标",
            "home": "任务主入口或可编辑入口",
            "search_entry": "可编辑区域（搜索框/输入框/文本框）",
            "search_input": "可编辑区域本体",
            "submit_search": "执行按钮（搜索/确认/提交/执行）",
            "results": "目标结果项或下一步功能入口",
            "detail": "目标功能控件、可编辑区域或任务相关按钮",
            "confirm": "最终执行按钮（发送/提交/确认/发布）",
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
