#!/usr/bin/env python3
"""
GLM-4V 验证测试 - 使用 test_runner 的验证逻辑，但输出详细对比
"""

import os
import sys
import json
import base64
import io

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

def encode_image_to_base64(image: Image.Image) -> str:
    """将图片编码为 base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def format_action(action: str, params: Dict) -> str:
    """格式化动作显示"""
    if not action:
        return "N/A"
    action = action.upper()
    if action == "CLICK":
        point = params.get("point", [])
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

def check_click(pred_params: Dict, ref_params: Dict) -> Tuple[bool, str]:
    """验证 CLICK - 使用归一化坐标"""
    point = pred_params.get('point')
    if not point or len(point) != 2:
        return False, "坐标无效"
    
    x, y = point
    x_min, x_max = ref_params['x']
    y_min, y_max = ref_params['y']
    
    if x_min < x < x_max and y_min < y < y_max:
        return True, "坐标匹配"
    else:
        return False, f"坐标越界"

def check_type(pred_params: Dict, ref_params: Dict) -> Tuple[bool, str]:
    """验证 TYPE"""
    import re
    pred_text = pred_params.get("text", "")
    ref_text = ref_params.get("text", "")
    
    if ref_text.startswith(".*") or ref_text.endswith(".*"):
        pattern = ref_text
        if re.search(pattern, pred_text):
            return True, "正则匹配"
        else:
            return False, "正则不匹配"
    else:
        if pred_text == ref_text:
            return True, "精确匹配"
        else:
            return False, "文本不匹配"

def check_open(pred_params: Dict, ref_params: Dict) -> Tuple[bool, str]:
    """验证 OPEN"""
    pred_app = pred_params.get("app_name", "")
    ref_app = ref_params.get("app", "")
    if pred_app == ref_app:
        return True, "应用匹配"
    else:
        return False, "应用不匹配"

def check_complete(pred_params: Dict, ref_params: Dict) -> Tuple[bool, str]:
    """验证 COMPLETE"""
    return True, "OK"

def verify_action(pred_action: str, pred_params: Dict, ref_action: str, ref_params: Dict) -> Tuple[bool, str]:
    """验证预测结果与参考是否匹配"""
    if pred_action.upper() != ref_action.upper():
        return False, f"动作不匹配"
    
    action = pred_action.upper()
    if action == "CLICK":
        return check_click(pred_params, ref_params)
    elif action == "TYPE":
        return check_type(pred_params, ref_params)
    elif action == "OPEN":
        return check_open(pred_params, ref_params)
    elif action == "COMPLETE":
        return check_complete(pred_params, ref_params)
    elif action == "SCROLL":
        return True, "SCROLL"
    
    return False, f"未知动作"

def test_case(case_name: str):
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
    
    # 获取初始状态
    first_status = None
    for key in ref_data.keys():
        if key not in ["case_overview"] and not key.startswith("_"):
            first_status = key
            break
    
    current_status = first_status
    step_count = 1
    
    # 历史记录
    history_messages = []
    history_actions = []
    
    print(f"{'步骤':<5} {'状态':<6} {'结果':<4} {'模型预测':<40} {'官方参考':<40} {'说明':<20}")
    print("-" * 130)
    
    correct_steps = 0
    total_steps = 0
    
    while current_status != '#' and step_count <= 30:
        # 获取截图
        screenshot_path = test_dir / f"{current_status}.png"
        if not screenshot_path.exists():
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
            break
        
        # 尝试所有可能的参考路径
        is_valid = False
        matched_ref = None
        message = ""
        next_status = "#"
        
        for ref_entry in ref_entry_list:
            ref_action = ref_entry.get("action", "")
            ref_params = ref_entry.get("params", {})
            
            valid, msg = verify_action(
                agent_output.action, agent_output.parameters,
                ref_action, ref_params
            )
            
            if valid:
                is_valid = True
                matched_ref = ref_entry
                message = msg
                next_status = ref_entry.get("next", "#")
                break
        
        # 如果没有匹配，使用第一个参考
        if not matched_ref:
            matched_ref = ref_entry_list[0]
            ref_action = matched_ref.get("action", "")
            ref_params = matched_ref.get("params", {})
            _, message = verify_action(
                agent_output.action, agent_output.parameters,
                ref_action, ref_params
            )
            next_status = matched_ref.get("next", "#")
        
        total_steps += 1
        if is_valid:
            correct_steps += 1
            status_icon = "✅"
        else:
            status_icon = "❌"
        
        # 格式化显示
        pred_str = format_action(agent_output.action, agent_output.parameters)
        ref_str = format_action(matched_ref.get("action", ""), matched_ref.get("params", {}))
        
        print(f"{step_count:<5} {current_status:<6} {status_icon:<4} {pred_str:<40} {ref_str:<40} {message:<20}")
        
        # 更新历史 (与 test_runner 一致)
        screenshot_base64 = encode_image_to_base64(screenshot)
        history_messages.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": screenshot_base64}}]
        })
        history_messages.append({
            "role": "assistant",
            "content": f"Action: {agent_output.action}"
        })
        history_actions.append({
            "step": step_count,
            "action": agent_output.action,
            "parameters": agent_output.parameters,
            "is_valid": is_valid
        })
        
        # 更新状态
        current_status = next_status
        step_count += 1
    
    print("-" * 130)
    accuracy = correct_steps / total_steps if total_steps > 0 else 0
    passed = accuracy == 1.0
    result_text = "PASS" if passed else "FAIL"
    print(f"总结: {correct_steps}/{total_steps} 步正确 ({accuracy:.1%}) - {result_text}")
    
    return passed, correct_steps, total_steps

def main():
    """主函数"""
    test_cases = [
        "step_meituan_onekey_0001",
    ]
    
    print("\n" + "=" * 100)
    print("GLM-4V 验证测试 - 模型预测 vs 官方参考")
    print("=" * 100)
    
    for case in test_cases:
        test_case(case)

if __name__ == "__main__":
    main()
