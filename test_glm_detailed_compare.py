#!/usr/bin/env python3
"""
GLM-4V 详细对比测试 - 使用 test_runner 的完整逻辑，但显示详细对比
"""

import os
import sys
import json

# 设置环境变量
os.environ['VLM_API_KEY'] = '60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
os.environ['DEBUG_API_URL'] = 'https://open.bigmodel.cn/api/paas/v4'
os.environ['DEBUG_MODEL_ID'] = 'glm-4v-flash'

sys.path.insert(0, './submission/src')

from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
from agent import Agent
from agent_base import AgentInput

def load_ref_data(test_dir: Path) -> Dict:
    """加载参考数据"""
    ref_file = test_dir / "ref.json"
    with open(ref_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_initial_info(ref_data: Dict) -> Tuple[str, str, int]:
    """获取初始状态、指令和最大步数"""
    first_status = None
    for key in ref_data.keys():
        if key not in ["case_overview"] and not key.startswith("_"):
            first_status = key
            break
    
    instruction = ref_data.get("case_overview", {}).get("instruction", "")
    
    # 计算最大步数
    step_max = 0
    for key, value in ref_data.items():
        if key not in ["case_overview"] and not key.startswith("_"):
            if isinstance(value, list):
                step_max = max(step_max, int(key) if key.isdigit() else 0)
    step_max += 2
    
    return first_status, instruction, step_max

def format_action(action: str, params: Dict) -> str:
    """格式化动作显示"""
    if not action:
        return "N/A"
    action = action.upper()
    if action == "CLICK":
        point = params.get("point", [])
        region = params.get("_candidate_region", "")
        if region:
            return f"CLICK {point} ({region})"
        return f"CLICK {point}"
    elif action == "TYPE":
        text = params.get("text", "")
        return f'TYPE "{text}"'
    elif action == "SCROLL":
        start = params.get("start_point", [])
        end = params.get("end_point", [])
        return f"SCROLL {start}->{end}"
    elif action == "OPEN":
        app = params.get("app_name", "")
        return f'OPEN "{app}"'
    elif action == "COMPLETE":
        return "COMPLETE"
    return f"{action} {params}"

def check_result(ref_data: Dict, current_status: str, pred_action: str, 
                 pred_params: Dict, width: int, height: int) -> Tuple[bool, str]:
    """验证预测结果与参考是否匹配"""
    ref_entry_list = ref_data.get(current_status, [])
    if not ref_entry_list:
        return False, "无参考数据"
    
    ref_entry = ref_entry_list[0]
    ref_action = ref_entry.get("action", "").upper()
    ref_params = ref_entry.get("params", {})
    
    # 动作类型不匹配
    if pred_action.upper() != ref_action:
        return False, f"动作不匹配: 预测={pred_action}, 参考={ref_action}"
    
    # 根据动作类型验证参数
    if ref_action == "CLICK":
        # 获取预测坐标
        pred_point = pred_params.get("point", [])
        if not pred_point or len(pred_point) < 2:
            return False, "CLICK 坐标无效"
        
        # 获取参考坐标范围
        x_range = ref_params.get("x_real", [])
        y_range = ref_params.get("y_real", [])
        
        if len(x_range) >= 2 and len(y_range) >= 2:
            x_min, x_max = x_range[0], x_range[1]
            y_min, y_max = y_range[0], y_range[1]
            x, y = pred_point[0], pred_point[1]
            
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False, f"坐标越界: ({x},{y}) 不在 ([{x_min},{x_max}], [{y_min},{y_max}])"
    
    elif ref_action == "TYPE":
        pred_text = pred_params.get("text", "")
        ref_text = ref_params.get("text", "")
        
        # 检查是否是正则匹配
        import re
        if ref_text.startswith(".*") or ref_text.endswith(".*"):
            pattern = ref_text
            if not re.search(pattern, pred_text):
                return False, f"文本不匹配: 预测='{pred_text}', 正则='{pattern}'"
        else:
            if pred_text != ref_text:
                return False, f"文本不匹配: 预测='{pred_text}', 参考='{ref_text}'"
    
    elif ref_action == "OPEN":
        pred_app = pred_params.get("app_name", "")
        ref_app = ref_params.get("app", "")
        if pred_app != ref_app:
            return False, f"应用不匹配: 预测='{pred_app}', 参考='{ref_app}'"
    
    return True, "匹配"

def test_case(case_name: str) -> Tuple[bool, int, int]:
    """测试单个用例"""
    test_dir = Path(f"./test_data/offline/{case_name}")
    
    print("\n" + "=" * 100)
    print(f"测试用例: {case_name}")
    print("=" * 100)
    
    # 加载参考数据
    ref_data = load_ref_data(test_dir)
    instruction = ref_data.get("case_overview", {}).get("instruction", "")
    print(f"指令: {instruction}")
    print()
    
    # 初始化 Agent
    agent = Agent()
    agent.reset()
    
    # 获取初始信息
    first_status, instruction, step_max = get_initial_info(ref_data)
    current_status = first_status
    step_count = 1
    
    # 历史记录
    history_messages = []
    history_actions = []
    
    print(f"{'步骤':<5} {'状态':<6} {'结果':<4} {'模型预测':<50} {'官方参考':<50} {'说明':<20}")
    print("-" * 150)
    
    correct_steps = 0
    total_steps = 0
    
    while current_status != '#' and step_count <= min(step_max, 30):
        # 获取截图
        screenshot_path = test_dir / f"{current_status}.png"
        if not screenshot_path.exists():
            print(f"步骤 {step_count}: 截图不存在 {screenshot_path}")
            break
        
        screenshot = Image.open(screenshot_path)
        
        # 创建 AgentInput
        agent_input = AgentInput(
            instruction=instruction,
            current_image=screenshot,
            step_count=step_count,
            history_messages=history_messages,
            history_actions=history_actions
        )
        
        # 调用 Agent
        try:
            agent_output = agent.act(agent_input)
        except Exception as e:
            print(f"步骤 {step_count}: Agent 调用失败 - {e}")
            break
        
        # 获取参考数据
        ref_entry_list = ref_data.get(current_status, [])
        if not ref_entry_list:
            print(f"步骤 {step_count}: 无参考数据")
            break
        
        ref_entry = ref_entry_list[0]
        ref_action = ref_entry.get("action", "")
        ref_params = ref_entry.get("params", {})
        next_status = ref_entry.get("next", "#")
        
        # 验证结果
        is_valid, message = check_result(
            ref_data, current_status, 
            agent_output.action, agent_output.parameters,
            screenshot.width, screenshot.height
        )
        
        total_steps += 1
        if is_valid:
            correct_steps += 1
            status_icon = "✅"
        else:
            status_icon = "❌"
        
        # 格式化显示
        pred_str = format_action(agent_output.action, agent_output.parameters)
        ref_str = format_action(ref_action, ref_params)
        
        print(f"{step_count:<5} {current_status:<6} {status_icon:<4} {pred_str:<50} {ref_str:<50} {message:<20}")
        
        # 显示模型原始输出（简化）
        raw_short = agent_output.raw_output[:80].replace('\n', ' ') if agent_output.raw_output else ""
        if len(agent_output.raw_output or "") > 80:
            raw_short += "..."
        print(f"      模型输出: {raw_short}")
        
        # 更新历史
        history_actions.append({
            "step": step_count,
            "action": agent_output.action,
            "parameters": agent_output.parameters,
            "is_valid": is_valid
        })
        
        # 更新状态
        current_status = next_status
        step_count += 1
    
    print("-" * 150)
    accuracy = correct_steps / total_steps if total_steps > 0 else 0
    passed = accuracy == 1.0
    result_text = "PASS" if passed else "FAIL"
    print(f"总结: {correct_steps}/{total_steps} 步正确 ({accuracy:.1%}) - {result_text}")
    
    return passed, correct_steps, total_steps

def main():
    """主函数"""
    test_cases = [
        "step_meituan_onekey_0001",
        "step_douyin_onekey_0008", 
        "step_baidumap_onekey_0010",
    ]
    
    print("\n" + "=" * 100)
    print("GLM-4V 详细对比测试 - 模型预测 vs 官方参考")
    print("=" * 100)
    
    total_pass = 0
    total_correct = 0
    total_steps = 0
    
    for case in test_cases:
        passed, correct, steps = test_case(case)
        if passed:
            total_pass += 1
        total_correct += correct
        total_steps += steps
    
    print("\n" + "=" * 100)
    print("总结果")
    print("=" * 100)
    print(f"通过用例: {total_pass}/{len(test_cases)}")
    print(f"总步骤: {total_correct}/{total_steps} ({total_correct/total_steps:.1%})" if total_steps > 0 else "N/A")

if __name__ == "__main__":
    main()
