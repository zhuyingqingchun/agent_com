"""
Prompt 工程模块 - 针对 GUI 控制任务优化
"""
from typing import Dict, List, Any


class PromptTemplate:
    """Prompt 模板基类"""

    @staticmethod
    def get_system_prompt() -> str:
        raise NotImplementedError

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any] = None,
        history: List[Dict[str, Any]] = None,
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        return f"""请从以下指令中提取所有关键信息：

{instruction}

输出 JSON：""".strip()
    @staticmethod
    def get_system_prompt() -> str:
        return """你是任务解析助手。从用户指令中提取关键信息，输出 JSON。

【输出格式】
{
  "app_name": "应用名称，如美团、百度地图、抖音",
  "shop_name": "店铺/商家名称",
  "product_name": "商品/菜品名称",
  "search_query": "完整搜索关键词",
  "task_type": "任务类型：food/map/video/search/shopping"
}

【提取规则】
1. 应用名：从"打开XX"、"去XX"、"在XX"中提取应用名称
2. 店铺名：提取"XX店铺"、"XX店"、"XX商家"中的名称
3. 商品名：提取要购买的具体商品/菜品名称
4. 搜索词：完整的搜索内容，包含店铺和商品
5. 只输出 JSON，不要解释

【示例】
指令：去美团外卖购买窑村干锅猪蹄（科技大学店）店铺的干锅排骨
输出：{
  "app_name": "美团",
  "shop_name": "窑村干锅猪蹄（科技大学店）",
  "product_name": "干锅排骨",
  "search_query": "窑村干锅猪蹄（科技大学店）干锅排骨",
  "task_type": "food"
}""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any] = None,
        history: List[Dict[str, Any]] = None,
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        return f"""请从以下指令中提取所有关键信息：

{instruction}

输出 JSON：""".strip()


class ActionPrompt(PromptTemplate):
    """
    动作执行 Prompt - 第二阶段：根据截图执行动作
    """

    @staticmethod
    def get_system_prompt() -> str:
        return """你是安卓 UI 自动化助手。根据截图和任务信息，输出 JSON 动作。

【动作格式】
OPEN: {"action":"OPEN","parameters":{"app_name":"应用名"}}
CLICK: {"action":"CLICK","parameters":{"point":[x,y]}} 坐标[0-1000]
TYPE: {"action":"TYPE","parameters":{"text":"内容"}}
COMPLETE: {"action":"COMPLETE","parameters":{}}

【关键坐标】
- 顶部搜索框: [500, 110]
- 右上角搜索: [930, 110]
- 左上角返回: [88, 110]

【重要规则】
1. 只输出 JSON，不要任何其他文字
2. 搜索框在顶部 y≈100
3. 输入前必须先点击搜索框
4. 看到目标时输出 COMPLETE
5. TYPE 时必须使用提供的【搜索词】，不要输出"action"或其他占位符""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        slots = state.get("slots", {})
        app_name = state.get("app_name", "")
        query = slots.get('query', '')
        shop_name = slots.get('shop_name', '')
        product_name = slots.get('product_name', '')
        
        # 历史记录
        history_str = ""
        if history:
            actions = [h.get('action', '') for h in history[-3:]]
            history_str = f"已执行: {' → '.join(actions)}"
        
        # 构建完成目标提示
        completion_targets = []
        if shop_name:
            completion_targets.append(f"店铺：{shop_name}")
        if product_name:
            completion_targets.append(f"商品：{product_name}")
        if query and not shop_name and not product_name:
            completion_targets.append(f"搜索：{query}")
        completion_hint = f"【完成目标】{'；'.join(completion_targets)}" if completion_targets else ""
        
        # 阶段指导 - 明确告诉模型要做什么
        if phase == "launch":
            phase_guide = f"第一步：打开应用 '{app_name}'"
            action_hint = f'输出: OPEN {{"app_name":"{app_name}"}}'
        elif phase == "home":
            phase_guide = "第二步：点击顶部搜索框"
            action_hint = '输出: CLICK {"point":[500,110]}'
        elif phase == "search_entry":
            phase_guide = "第三步：点击搜索框激活输入"
            action_hint = '输出: CLICK {"point":[500,110]}'
        elif phase == "search_input":
            phase_guide = f"第四步：输入搜索词"
            action_hint = f'输出: TYPE {{"text":"{query}"}}'
        elif phase == "submit_search":
            phase_guide = "第五步：点击搜索按钮"
            action_hint = '输出: CLICK {"point":[930,110]}'
        elif phase == "results":
            phase_guide = "第六步：点击搜索结果"
            action_hint = "看截图找到目标，输出 CLICK"
        elif phase == "detail":
            phase_guide = "第七步：确认目标已显示"
            action_hint = "输出: COMPLETE"
        else:
            phase_guide = "执行下一步"
            action_hint = "根据截图判断"

        return f"""【任务】{instruction}
【应用】{app_name}
【搜索词】{query}
{completion_hint}
{history_str}
【当前步骤】{phase_guide}
【应该输出】{action_hint}

看截图，输出 JSON：""".strip()


class GroundedActionPrompt(PromptTemplate):
    """
    更适合视觉模型的动作 Prompt。

    目标：
    1. 避免把固定模板词当成输出内容
    2. 避免把坐标写死成 [500,110] / [930,110]
    3. 强化“不要跳步”和“不要过早 COMPLETE”
    """

    @staticmethod
    def get_system_prompt() -> str:
        return """你是安卓 GUI 自动化助手。你必须只根据当前截图、任务目标和最近历史，输出一步最合理的 JSON 动作。

【只允许输出一个 JSON 对象】
CLICK: {"action":"CLICK","parameters":{"point":[x,y]}}
TYPE: {"action":"TYPE","parameters":{"text":"内容"}}
OPEN: {"action":"OPEN","parameters":{"app_name":"应用名"}}
SCROLL: {"action":"SCROLL","parameters":{"start_point":[x1,y1],"end_point":[x2,y2]}}
COMPLETE: {"action":"COMPLETE","parameters":{}}

【CLICK 可选增强字段】
如果你能判断点击目标区域，请在最外层额外输出:
"candidate_region":"TOP_SEARCH_BOX" 或 "TOP_RIGHT_ICON" 等
可选值仅限:
TOP_SEARCH_BOX, TOP_RIGHT_ICON, TOP_RIGHT_SMALL, TOP_LEFT_ICON, MID_LIST, CENTER_PANEL, BOTTOM_BAR

【硬性规则】
1. 只能输出 JSON，不要解释，不要 markdown，不要额外字段。
2. 坐标必须依据截图中的真实位置，不要机械输出固定坐标。
3. 没有进入输入框之前，不要直接 TYPE。
4. 页面上仍然存在明显的“搜索框 / 搜索按钮 / 结果项 / 评论输入框 / 确认按钮”时，不要过早 COMPLETE。
5. OPEN 只用于任务刚开始、还没打开目标应用时。
6. 不要输出占位词，例如 "action"、"app"、"text"。
7. 不要因为任务里有关键词，就直接 TYPE；先确认输入框已经被点开。
8. 如果页面顶部右侧有明显的小搜索图标/放大镜，优先点它的中心，不要点顶部中间的空白处。
9. 如果页面底部右侧有明显的搜索入口/放大镜/发现按钮，优先点那个真实按钮。

【决策顺序】
先判断当前是否仍在桌面/启动页，需要 OPEN。
否则判断是否需要先点击搜索入口、搜索框、输入框、评论框等可编辑区域。
然后判断是否需要 TYPE。
然后判断是否需要点击搜索按钮、发送按钮、确认按钮、结果项。
只有任务目标已经明显达成时才输出 COMPLETE。""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        app_name = state.get("app_name", "")
        task_type = state.get("task_type", "generic")
        typed_texts = state.get("typed_texts", [])

        history_lines = []
        for item in history[-3:]:
            action = item.get("action", "")
            params = item.get("parameters", {})
            history_lines.append(f"- {action} {params}")
        history_text = "\n".join(history_lines) if history_lines else "- 无"

        return f"""【任务指令】
{instruction}

【目标应用】
{app_name or "从任务中判断"}

【已执行步数】
{state.get('step_count', '?')}

【最近动作】
{history_text}

【已输入过的文本】
{typed_texts if typed_texts else "无"}

【输出偏好】
- 对 CLICK，尽量同时给出 point 和 candidate_region。
- candidate_region 只是辅助，不确定时可以不填，但不要填错应用名或占位词。
- 坐标必须依据截图中的真实位置，不要机械输出固定坐标。

现在只输出下一步动作 JSON：""".strip()


class OptimizedPrompt(PromptTemplate):
    """
    优化版 Prompt - 模型提取关键词
    
    核心优化点：
    1. 让模型从指令中提取应用名和关键词
    2. 精简 System Prompt，保留关键信息
    3. 强化 User Prompt 的当前步骤指导
    """

    @staticmethod
    def get_system_prompt() -> str:
        return """你是安卓 UI 自动化助手。根据截图和任务指令，输出 JSON 动作。

【动作格式】
OPEN: {"action":"OPEN","parameters":{"app_name":"应用名称"}}
CLICK: {"action":"CLICK","parameters":{"point":[x,y]}} 坐标范围 x[0-1000], y[0-1000]
TYPE: {"action":"TYPE","parameters":{"text":"要输入的内容"}}
COMPLETE: {"action":"COMPLETE","parameters":{}}

【关键词提取】
从用户指令中提取：
- 应用名：如"美团"、"百度地图"、"抖音"
- 搜索词：如"周杰伦"、"北京到上海"、"附近的火锅店"

【标准流程】
1. OPEN: 打开应用（第一步）
2. CLICK: 点击顶部搜索框 [500, 110]（第二步）
3. CLICK: 再次点击搜索框激活（第三步）
4. TYPE: 输入搜索词（第四步）
5. CLICK: 点击搜索按钮 [930, 110]（第五步）
6. CLICK: 点击搜索结果（第六步）
7. COMPLETE: 任务完成

【重要提示】
- 只输出 JSON，不要任何解释
- 搜索框通常在页面顶部 y≈100
- 输入前必须先点击搜索框
- 看到目标内容时输出 COMPLETE""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        slots = state.get("slots", {})
        app_name = state.get("app_name", "")
        query = slots.get('query', '')
        
        # 历史记录（简洁）
        history_str = ""
        if history:
            actions = [h.get('action', '') for h in history[-3:]]
            history_str = f"已执行: {' → '.join(actions)}"
        
        # 阶段指导
        phase_instructions = {
            "launch": "第一步：打开应用",
            "home": "第二步：点击顶部搜索框 [500, 110]",
            "search_entry": "第三步：点击搜索框激活输入",
            "search_input": "第四步：输入搜索词",
            "submit_search": "第五步：点击搜索按钮 [930, 110]",
            "results": "第六步：点击搜索结果",
            "detail": "第七步：确认目标已显示，输出 COMPLETE",
        }
        
        current_step = phase_instructions.get(phase, "根据截图判断下一步")

        return f"""【任务指令】{instruction}

【执行进度】{history_str or '开始'}
【当前步骤】{current_step}

请从指令中提取应用名和搜索词，看截图，输出下一步动作的 JSON：""".strip()


class TaskGuidedPrompt(PromptTemplate):
    """
    任务引导 Prompt - 针对具体任务流程优化
    强化任务分解和步骤引导
    """

    @staticmethod
    def get_system_prompt() -> str:
        return """你是安卓应用自动化助手。根据截图执行任务，输出 JSON 动作。

【动作格式】
OPEN: {"action":"OPEN","parameters":{"app_name":"应用名"}}
CLICK: {"action":"CLICK","parameters":{"point":[x,y]}} 坐标范围[0-1000]
TYPE: {"action":"TYPE","parameters":{"text":"内容"}}
COMPLETE: {"action":"COMPLETE","parameters":{}}

【关键坐标】
- 顶部搜索框: [500, 110]
- 左上角返回: [88, 110]
- 右上角搜索: [930, 110]
- 屏幕中央: [500, 500]
- 底部导航: [500, 900]

【任务流程】
1. 打开应用
2. 找到并点击搜索框（顶部 y≈100）
3. 输入搜索关键词
4. 点击搜索按钮
5. 在结果中选择目标
6. 完成任务

【规则】
- 只输出 JSON，不要解释
- 搜索框通常在页面顶部
- 输入前必须先点击搜索框
- 看到目标内容时输出 COMPLETE""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        slots = state.get("slots", {})
        
        # 提取关键词
        query = slots.get('query', '')
        
        # 历史动作摘要
        history_summary = ""
        if history:
            last = history[-1]
            action = last.get('action', '')
            params = last.get('parameters', {})
            if action == 'CLICK':
                point = params.get('point', [])
                history_summary = f"上一步: 点击 {point}"
            elif action == 'TYPE':
                text = params.get('text', '')
                history_summary = f"上一步: 输入 '{text}'"
            elif action == 'OPEN':
                app = params.get('app_name', '')
                history_summary = f"上一步: 打开 {app}"
            else:
                history_summary = f"上一步: {action}"

        # 根据阶段给出明确指示
        if phase == "launch":
            guide = "打开应用，开始任务"
        elif phase == "home":
            guide = "点击顶部搜索框 [500, 110] 进入搜索"
        elif phase == "search_entry":
            guide = f"点击搜索框，准备输入 '{query}'"
        elif phase == "search_input":
            guide = f"输入关键词: {query}"
        elif phase == "submit_search":
            guide = "点击搜索按钮或确认键执行搜索"
        elif phase == "results":
            guide = f"在结果列表中找到 '{query}' 并点击"
        elif phase == "detail":
            guide = "确认目标页面已加载，输出 COMPLETE"
        else:
            guide = "根据截图判断下一步操作"

        return f"""任务: {instruction}
{history_summary}
当前阶段: {phase}

下一步: {guide}

看截图，输出精确的 JSON 动作。""".strip()


class VisualGroundingPrompt(PromptTemplate):
    """
    视觉定位 Prompt - 强调基于截图的视觉理解
    """

    @staticmethod
    def get_system_prompt() -> str:
        return """分析截图，执行 GUI 操作。

【输出格式】
{"action":"CLICK","parameters":{"point":[x,y]}}
{"action":"TYPE","parameters":{"text":"内容"}}
{"action":"OPEN","parameters":{"app_name":"应用"}}
{"action":"COMPLETE","parameters":{}}

【坐标提示】
屏幕坐标范围 [0, 1000]：
- 顶部区域 y: 50-180（搜索框、标题栏）
- 中部区域 y: 180-800（内容列表）
- 底部区域 y: 800-1000（导航栏）

【操作逻辑】
1. 观察截图中的 UI 元素
2. 确定目标元素位置
3. 选择合适的坐标点击
4. 只输出 JSON""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        slots = state.get("slots", {})
        
        # 构建简洁的上下文
        context = f"阶段: {phase}"
        if slots.get('query'):
            context += f" | 关键词: {slots['query']}"
        
        return f"""{instruction}
{context}

看截图，找到目标元素，输出 JSON。""".strip()


class SimplePrompt(PromptTemplate):
    """简洁 Prompt"""

    @staticmethod
    def get_system_prompt() -> str:
        return """输出 JSON 动作。格式：
CLICK:{"action":"CLICK","parameters":{"point":[x,y]}}
TYPE:{"action":"TYPE","parameters":{"text":"内容"}}
OPEN:{"action":"OPEN","parameters":{"app_name":"应用"}}
COMPLETE:{"action":"COMPLETE","parameters":{}}""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        return f"""任务：{instruction}
阶段：{phase}
看截图，输出 JSON。""".strip()


class DetailedPrompt(PromptTemplate):
    """详细 Prompt"""

    @staticmethod
    def get_system_prompt() -> str:
        return """你是安卓 GUI 自动化 Agent。根据截图和任务指令，输出控制动作 JSON。

动作说明：
- CLICK: 点击屏幕位置，坐标 [0-1000]
- TYPE: 输入文本
- SCROLL: 滑动屏幕
- OPEN: 打开应用
- COMPLETE: 任务完成

只输出一个 JSON 对象，不要解释。""".strip()

    @staticmethod
    def get_user_prompt(
        instruction: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_hint: str = "",
        app_memory: str = "",
    ) -> str:
        phase = state.get("phase", "launch")
        slots = state.get("slots", {})
        
        history_details = []
        for i, h in enumerate(history[-3:], 1):
            action = h.get('action', '')
            params = h.get('parameters', {})
            if action == 'CLICK':
                point = params.get('point', [])
                history_details.append(f"{i}. 点击 {point}")
            elif action == 'TYPE':
                text = params.get('text', '')
                history_details.append(f"{i}. 输入 '{text}'")
            else:
                history_details.append(f"{i}. {action}")

        history_str = " | ".join(history_details) if history_details else "无"

        return f"""【任务】{instruction}
【应用】{state.get('app_name', '未知')}
【阶段】{phase}
【关键词】{slots.get('query', '无')}
【历史】{history_str}

分析截图，输出下一步动作 JSON。""".strip()


# Prompt 注册表
PROMPT_REGISTRY = {
    "optimized": OptimizedPrompt,          # 优化版（推荐）
    "action": ActionPrompt,                # 动作执行（第二阶段）
    "grounded_action": GroundedActionPrompt,  # 更稳的视觉动作 prompt
    "task_guided": TaskGuidedPrompt,       # 任务引导
    "visual_grounding": VisualGroundingPrompt,  # 视觉定位
    "simple": SimplePrompt,                # 简洁
    "detailed": DetailedPrompt,            # 详细
}


def get_prompt_template(name: str = "optimized") -> PromptTemplate:
    """获取指定的 prompt 模板

    Args:
        name: prompt 模板名称

    Returns:
        PromptTemplate 实例
    """
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt template: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]()


def list_prompt_templates() -> List[str]:
    """列出所有可用的 prompt 模板"""
    return list(PROMPT_REGISTRY.keys())
