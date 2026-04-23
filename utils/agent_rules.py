import re
from typing import Any, Dict

from agent_base import ACTION_CLICK, ACTION_TYPE
from utils.agent_config import KNOWN_APPS


def extract_app_name(instruction: str) -> str:
    for app_name in KNOWN_APPS:
        if app_name in instruction:
            return app_name
    return ""


def infer_task_type(instruction: str, app_name: str) -> str:
    if "酒店" in instruction or "民宿" in instruction:
        return "hotel"
    if "美食" in instruction or "餐厅" in instruction or "外卖" in instruction or app_name == "美团":
        return "food"
    if "车票" in instruction or "高铁" in instruction or "火车" in instruction:
        return "train"
    if "地图" in instruction or "导航" in instruction or "路线" in instruction or app_name == "百度地图":
        return "map"
    if "视频" in instruction or "番剧" in instruction or "直播" in instruction or app_name in {"抖音", "哔哩哔哩", "爱奇艺", "腾讯视频", "芒果TV", "快手"}:
        return "video"
    if "购买" in instruction or "下单" in instruction or "商品" in instruction:
        return "shopping"
    if "搜索" in instruction or "查找" in instruction or "看看" in instruction:
        return "search"
    return "generic"


def extract_slots(instruction: str, task_type: str) -> Dict[str, str]:
    """提取任务关键信息（搜索关键词、目的地等）"""
    slots: Dict[str, str] = {}
    
    # 1. 提取引号内的内容（最精确的关键词）
    quoted = re.findall(r'["""](.*?)["""]', instruction)
    if quoted:
        slots["query"] = quoted[0].strip()
        return slots  # 引号内内容最优先
    
    # 2. 提取"为"、"成"后面的目标（如"更换语音包为XXX"）
    # 优先匹配"为XXX"或"成XXX"格式
    wei_match = re.search(r"(?:为|成|换成|改成|设为|设置为)([^，。！？,]+?)(?:，|,|。|！|？|$)", instruction)
    if wei_match:
        value = wei_match.group(1).strip(" '").strip()
        # 过滤掉常见无效值
        if len(value) >= 1 and value not in ["默认", "当前", "现有", "新的"]:
            slots["query"] = value[:30]
            return slots
    
    # 3. 提取"设置XXX语音包/主题"格式中的XXX
    setting_match = re.search(r"(?:设置|选择|使用|更换)(.+?)(?:语音包|铃声|主题|皮肤|头像|昵称|名称)", instruction)
    if setting_match:
        value = setting_match.group(1).strip(" '").strip()
        if len(value) >= 1 and value not in ["默认", "当前"]:
            slots["query"] = value[:30]
            return slots
    
    # 3. 提取店铺/商品名称（在"购买"、"下单"、"买"之后，"店铺"、"的"之前）
    shop_patterns = [
        r"(?:购买|买|下单)(.+?)(?:店铺|的|，|,|。)",  # 购买XXX店铺/的
        r"(?:去|在)(.+?)(?:购买|买|下单)",  # 去XXX购买
    ]
    for pattern in shop_patterns:
        m = re.search(pattern, instruction)
        if m:
            value = m.group(1).strip(" '").strip()
            # 清理常见前缀
            value = re.sub(r"^(?:美团外卖|美团|淘宝|京东|拼多多)", "", value).strip()
            if len(value) >= 2:
                slots["query"] = value[:30]  # 限制长度
                return slots
    
    # 4. 标准搜索模式
    patterns = [
        (r"帮我搜(?:索)?(.+)", "query"),
        (r"查一下(.+)", "query"),
        (r"看看(.+)", "query"),
        (r"找一下(.+)", "query"),
        (r"搜索(.+)", "query"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, instruction)
        if m and key not in slots:
            value = m.group(1).strip("。！？!?，, ")
            if value:
                slots[key] = value[:30]
                return slots
    
    # 5. 城市/目的地提取
    city_match = re.search(r"去([^\s，。!?]{2,8})(?:市|县|镇)?", instruction)
    if city_match and "destination" not in slots:
        slots["destination"] = city_match.group(1)
    
    # 6. 附近地点
    near_match = re.search(r"(.+?)附近", instruction)
    if near_match and "location" not in slots:
        slots["location"] = near_match.group(1).strip()
    
    # 7. 任务类型特定提取
    if task_type == "shopping" and "query" not in slots:
        m = re.search(r"(?:买|购买|下单)(.+?)(?:，|,|。|！|？|$)", instruction)
        if m:
            slots["query"] = m.group(1).strip("。！？!?，, ")[:30]
    
    if task_type in {"video", "search"} and "query" not in slots:
        m = re.search(r"(?:搜索|查找|看看)(.+?)(?:，|,|。|！|？|$)", instruction)
        if m:
            slots["query"] = m.group(1).strip("。！？!?，, ")[:30]
    
    # 8. 兜底：提取核心名词短语
    if "query" not in slots:
        # 移除常见动词和介词，提取剩余的核心内容
        core = instruction.strip()
        # 移除前缀
        core = re.sub(r"^(?:帮我|请|去|在|打开|使用)", "", core).strip()
        # 移除后缀
        core = re.sub(r"(?:，|,|。|！|？).*$", "", core).strip()
        # 如果还很长，取前20个字符
        if len(core) > 20:
            # 尝试找到第一个动词后的内容
            verb_match = re.search(r"(?:买|购买|搜索|查找|去|看|换|设置)[了过]?(.+)", core)
            if verb_match:
                core = verb_match.group(1).strip()
        slots["query"] = core[:30] if core else instruction[:30]
    
    return slots


def decompose_instruction(instruction: str) -> Dict[str, Any]:
    result = {
        "app": "",
        "store_name": "",
        "product_name": "",
        "action_type": "",
        "raw": instruction,
    }
    app = extract_app_name(instruction)
    if app:
        result["app"] = app
    store_patterns = [
        r"([^，,（(]+?\([^)]*店[^)]*\))",
        r"(?:进入|搜索|去|到)[^，,。]*?([^\s,，。]+?店)",
        r"([^\s,，（(]{2,20}?[店铺馆楼])",
    ]
    for p in store_patterns:
        m = re.search(p, instruction)
        if m:
            result["store_name"] = m.group(1).strip()
            break
    product_patterns = [
        r"(?:店铺的|店的)[^\s,，]*?(.+?)(?:[,，]|地址|默认|选择|$)",
        r"(?:购买|买[^\s]*?|下单|点[^\s]*?)[\s]*「?([^\s,，。「」]+?)」?(?:[,，]|地址|默认|选择|$)",
        r"搜索[为找]*([^\s,，。]{2,15})",
        r"(?:播放|看|搜)([^\s,，。]+(?:视频|番剧|电影|剧集))",
    ]
    for p in product_patterns:
        m = re.search(p, instruction)
        if m and len(m.group(1).strip()) >= 2:
            candidate = m.group(1).strip()
            if candidate != result.get("store_name", ""):
                result["product_name"] = candidate
                break
    if "购买" in instruction or "下单" in instruction or "外卖" in instruction or "点" in instruction:
        result["action_type"] = "buy"
    elif "搜索" in instruction or "查找" in instruction:
        result["action_type"] = "search"
    elif "播放" in instruction or "看" in instruction:
        result["action_type"] = "play"
    elif "导航" in instruction or "路线" in instruction:
        result["action_type"] = "navigate"
    return result


def get_current_subgoal(instruction: str, state: Dict[str, Any], input_data: Any = None) -> str:
    parsed = decompose_instruction(instruction)
    phase = state.get("phase", "launch")
    typed_texts = state.get("typed_texts", [])
    step_count = getattr(input_data, "step_count", 0) if input_data else state.get("step_count", 0)
    app = parsed["app"] or state.get("app_name", "")
    store = parsed["store_name"]
    product = parsed["product_name"]
    action = parsed["action_type"]
    if phase == "launch":
        return f"打开{app}" if app else "打开目标应用"
    if phase in {"home"}:
        return f"在{app}中找到搜索入口或外卖入口"
    if phase in {"search_entry", "search_input"}:
        if store:
            return f"搜索并进入店铺「{store}」"
        elif product:
            return f"搜索「{product}」"
        return "搜索任务相关内容"
    if phase == "submit_search":
        return "点击搜索按钮确认搜索"
    if phase == "results":
        if store and not typed_texts:
            return f"点击搜索结果中的「{store}」进入店铺"
        return "点击正确的搜索结果项"
    if phase == "detail":
        if product and product not in "".join(typed_texts):
            return f"在店铺内搜索「{product}」"
        if action == "buy":
            return "选择商品规格/数量，加入购物车或确认下单"
        return "执行下一步操作（选择、确认等）"
    if phase == "confirm":
        return "确认订单/提交操作（注意地址是否需要选择）"
    return ""


def workflow_hint(app_name: str, task_type: str) -> str:
    return ""


def infer_page_type(state: Dict[str, Any], input_data: Any) -> str:
    phase = state.get("phase", "launch")
    if input_data.step_count == 1 and phase == "launch":
        return "home_or_launcher"
    last_action = state.get("last_action", "")
    if last_action == ACTION_TYPE:
        return "search_confirm"
    if state.get("search_box_clicked") and not state.get("typed_texts"):
        return "search_page"
    if state.get("typed_texts"):
        return "results_or_detail"
    if phase in {"confirm", "detail"}:
        return "detail"
    if phase in {"search_entry", "search_input", "submit_search"}:
        return "search_page"
    return "generic_page"


def allow_type_now(state: Dict[str, Any]) -> bool:
    return True


def verify_complete(state: Dict[str, Any], input_data: Any, raw_text: str) -> bool:
    return True


def should_force_complete(state: Dict[str, Any], input_data: Any, predicted_action: str) -> bool:
    return False


def looks_like_interrupt_needed(state: Dict[str, Any], page_stuck: bool, raw_text: str, input_data: Any) -> bool:
    last_repeat = state.get("repeat_action_count", 0) >= 1
    interrupt_tokens = ["关闭", "跳过", "以后再说", "取消", "知道了", "我知道了", "暂不", "不了", "稍后"]
    hit_text = any(tok in raw_text for tok in interrupt_tokens)
    return bool(page_stuck and (last_repeat or hit_text or input_data.step_count >= 3 and state.get("phase_retry_count", 0) >= 1))


def transition_phase(state: Dict[str, Any], target_phase: str, phase_transitions: Dict[str, str]) -> None:
    old_phase = state.get("phase", "launch")
    if target_phase != old_phase:
        state["phase_retry_count"] = 0
    state["phase"] = target_phase
    state["milestone"] = phase_transitions.get(target_phase, target_phase)
    if target_phase in {"launch", "home", "search_entry"}:
        state["complete_ready"] = False
    if target_phase == "search_input":
        state["submit_ready"] = False
    if target_phase in {"results", "detail", "confirm"}:
        state["complete_ready"] = True
