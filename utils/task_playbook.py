"""
通用任务流程先验记忆（Playbook）
- 根据指令自动匹配任务类型
- 按需加载对应的标准操作流程（非硬性规则，仅作为参考）
- 模型判断不需要时可不加载任何流程
"""
import re
from typing import Dict, List, Optional, Any


class TaskPlaybook:
    """任务流程先验记忆 — 可选择加载的流程模板"""
    
    _PLAYBOOK = {
        "takeout_order": {
            "name": "外卖下单",
            "keywords": ["购买", "下单", "点", "外卖", "店铺", "菜品"],
            "app_patterns": ["美团", "饿了么", "外卖"],
            "standard_flow": [
                ("open_app", "打开目标应用"),
                ("enter_channel", "进入外卖/点餐频道"),
                ("input_activate_1", "【激活】点击搜索框"),
                ("input_type_1", "【输入】TYPE 店铺名称"),
                ("input_confirm_1", "【确认】点击搜索按钮或结果项"),
                ("enter_store", "进入目标店铺页面"),
                ("input_activate_2", "【激活】点击店内搜索框"),
                ("input_type_2", "【输入】TYPE 具体商品/菜品名称"),
                ("input_confirm_2", "【确认】点击商品结果项"),
                ("select_spec", "选择规格/数量"),
                ("checkout", "点击去结算"),
                ("confirm_addr", "确认收货地址"),
                ("submit_order", "提交订单"),
            ],
            "tips": [
                "外卖通常需要两次搜索：先搜店铺名，进店后再搜具体商品",
                "店铺页面顶部有独立的店内搜索框，优先用它而不是左侧分类栏",
                "手机端输入必须遵循：激活→输入→确认 三步链路",
            ],
        },
        
        "map_navigation": {
            "name": "地图导航",
            "keywords": ["导航", "去", "路线", "怎么走", "到"],
            "app_patterns": ["百度地图", "高德地图", "地图"],
            "standard_flow": [
                ("open_app", "打开地图应用"),
                ("input_activate", "【激活】点击搜索框/目的地输入框"),
                ("input_type", "【输入】TYPE 目的地名称"),
                ("input_confirm", "【确认】点击搜索结果中的目标地点"),
                ("click_navigate", "点击导航/路线按钮"),
                ("select_mode", "选择出行方式（驾车/步行/公交）"),
                ("start_nav", "开始导航"),
                ("complete", "COMPLETE"),
            ],
            "tips": [
                "地图搜索后通常需要从多个结果中选择正确的目的地",
                "导航前需要确认起点和终点是否正确",
            ],
        },
        
        "video_search": {
            "name": "视频搜索播放",
            "keywords": ["看", "播放", "搜索视频", "找视频", "追剧"],
            "app_patterns": ["爱奇艺", "优酷", "哔哩哔哩", "B站", "腾讯视频"],
            "standard_flow": [
                ("open_app", "打开视频应用"),
                ("input_activate", "【激活】点击搜索框"),
                ("input_type", "【输入】TYPE 视频名称/关键词"),
                ("input_confirm", "【确认】点击搜索按钮"),
                ("select_video", "从结果列表中点击目标视频"),
                ("complete", "COMPLETE"),
            ],
            "tips": [
                "视频搜索可能需要区分剧集、电影、综艺等类型",
                "部分应用需要VIP才能观看某些内容",
            ],
        },
        
        "shopping": {
            "name": "购物下单",
            "keywords": ["买", "购物", "加购", "商品", "商品详情"],
            "app_patterns": ["淘宝", "京东", "拼多多", "购物"],
            "standard_flow": [
                ("open_app", "打开购物应用"),
                ("input_activate", "【激活】点击搜索框"),
                ("input_type", "【输入】TYPE 商品名称"),
                ("input_confirm", "【确认】点击搜索或结果项"),
                ("enter_detail", "进入商品详情页"),
                ("select_spec", "选择规格/参数/颜色等"),
                ("add_cart_or_buy", "加入购物车 或 立即购买"),
                ("confirm_order", "确认订单信息"),
                ("submit_order", "提交订单"),
            ],
            "tips": [
                "购物通常需要选择规格、颜色、尺码等属性",
                "注意区分「加入购物车」和「立即购买」两种操作",
            ],
        },
    }
    
    @classmethod
    def classify(cls, instruction: str, app_name: str = "") -> Optional[Dict[str, Any]]:
        """
        根据指令和应用名判断任务类型。
        返回匹配的流程模板，或 None（表示无法分类，不加载任何流程）。
        """
        instruction_lower = instruction.lower()
        
        best_match = None
        best_score = 0
        
        for task_type, config in cls._PLAYBOOK.items():
            score = 0
            
            for kw in config["keywords"]:
                if kw in instruction:
                    score += 2
            
            if app_name and any(p in app_name for p in config["app_patterns"]):
                score += 3
            
            if score > best_score:
                best_score = score
                best_match = (task_type, config)
        
        if best_score >= 2 and best_match:
            task_type, config = best_match
            return {
                "type": task_type,
                "name": config["name"],
                "flow": config["standard_flow"],
                "tips": config["tips"],
            }
        
        return None
    
    @classmethod
    def get_flow_display(cls, flow_info: Dict[str, Any], completed_count: int) -> str:
        """
        将流程模板格式化为 prompt 中的展示文本。
        completed_count: 已完成的步数（用于标记进度）
        """
        if not flow_info:
            return ""
        
        lines = [f"\n【标准流程参考 — {flow_info['name']}类任务】"]
        lines.append("（以下为该类任务的常见步骤，供参考，可根据实际情况调整）\n")
        
        flow = flow_info["flow"]
        for i, (step_id, step_desc) in enumerate(flow):
            num = i + 1
            if i < completed_count:
                lines.append(f"  {num}. ✅ {step_desc}")
            elif i == completed_count:
                lines.append(f"  {num}. ⬜ {step_desc}   ← 当前阶段")
            else:
                lines.append(f"  {num}. ○ {step_desc}")
        
        if flow_info.get("tips"):
            lines.append("\n【提示】")
            for tip in flow_info["tips"]:
                lines.append(f"  • {tip}")
        
        return "\n".join(lines)
    
    @classmethod
    def list_all_types(cls) -> List[str]:
        """列出所有支持的任务类型"""
        return [cfg["name"] for cfg in cls._PLAYBOOK.values()]


def classify_task(instruction: str, app_name: str = "") -> Optional[Dict[str, Any]]:
    """便捷函数：根据指令分类任务"""
    return TaskPlaybook.classify(instruction, app_name)


def get_task_flow_prompt(flow_info: Dict[str, Any], completed_steps: int) -> str:
    """便捷函数：获取流程的 prompt 展示文本"""
    return TaskPlaybook.get_flow_display(flow_info, completed_steps)


if __name__ == "__main__":
    test_cases = [
        ("去美团外卖购买窑村干锅猪蹄（科技大学店）店铺的干锅排骨", "美团"),
        ("用百度地图导航去天安门广场", "百度地图"),
        ("在爱奇艺搜索《狂飙》并播放", "爱奇艺"),
        ("在淘宝买一件衣服", "淘宝"),
        ("打开计算器算一下1+1", "计算器"),
    ]
    
    for inst, app in test_cases:
        result = classify_task(inst, app)
        if result:
            print(f"\n✅ [{result['name']}] {inst[:30]}...")
            print(TaskPlaybook.get_flow_display(result, 3))
        else:
            print(f"\n⚪ 未分类: {inst[:30]}... (不加载流程)")
