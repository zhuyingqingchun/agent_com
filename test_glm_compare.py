#!/usr/bin/env python3
"""
GLM-4V 详细对比测试 - 显示模型预测与官方参考的每一步比较
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image

# 设置环境变量
os.environ['VLM_API_KEY'] = '60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
os.environ['DEBUG_API_URL'] = 'https://open.bigmodel.cn/api/paas/v4'
os.environ['DEBUG_MODEL_ID'] = 'glm-4v-flash'

sys.path.insert(0, './submission/src')
from agent import Agent
from agent_base import AgentInput

def load_ref_data(test_dir: Path):
    """加载参考数据"""
    ref_file = test_dir / "ref.json"
    with open(ref_file, 'r') as f:
        return json.load(f)

def get_screenshot(test_dir: Path, status: str):
    """获取指定状态的截图"""
    screenshot_path = test_dir / f"{status}.png"
    if screenshot_path.exists():
        return Image.open(screenshot_path)
    return None

def format_action(action: str, params: dict) -> str:
    """格式化动作显示"""
    if action == "CLICK":
        point = params.get("point", [])
        return f"CLICK {point}"
    elif action == "TYPE":
        text = params.get("text", "")
        return f'TYPE "{text}"'
    elif action == "SCROLL":
        start = params.get("start_point", [])
        end = params.get("end_point", [])
        return f"SCROLL {start} -> {end}"
    elif action == "OPEN":
        app = params.get("app_name", "")
        return f'OPEN "{app}"'
    elif action == "COMPLETE":
        return "COMPLETE"
    return f"{action} {params}"

def compare_step(step_num: int, status: str, pred_action: str, pred_params: dict, 
                 ref_action_list: list, ref_params_list: list) -> tuple:
    """比较预测与参考"""
    # 简化比较逻辑
    pred_str = f"{pred_action.upper()}"
    ref_str = f"{ref_action_list[0].upper()}" if ref_action_list else "N/A"
    
    # 检查动作是否匹配
    action_match = pred_action.upper() == ref_action_list[0].upper() if ref_action_list else False
    
    return action_match, pred_str, ref_str

def test_case(case_name: str):
    """测试单个用例"""
    test_dir = Path(f"./test_data/offline/{case_name}")
    
    print("=" * 80)
    print(f"测试用例: {case_name}")
    print("=" * 80)
    
    # 加载参考数据
    ref_data = load_ref_data(test_dir)
    instruction = ref_data.get("case_overview", {}).get("instruction", "")
    print(f"指令: {instruction}")
    print()
    
    # 初始化 Agent
    agent = Agent()
    
    # 获取初始状态
    first_status = None
    for key in ref_data.keys():
        if key not in ["case_overview"] and not key.startswith("_"):
            first_status = key
            break
    
    if not first_status:
        print("错误: 未找到初始状态")
        return False, 0, 0
    
    current_status = first_status
    step_count = 1
    correct_steps = 0
    total_steps = 0
    
    print(f"{'步骤':<6} {'状态':<8} {'模型预测':<40} {'官方参考':<40} {'结果':<8}")
    print("-" * 120)
    
    while current_status != '#' and step_count <= 30:
        # 获取截图
        screenshot = get_screenshot(test_dir, current_status)
        if not screenshot:
            print(f"步骤 {step_count}: 未找到截图 {current_status}")
            break
        
        # 创建 AgentInput
        agent_input = AgentInput(
            instruction=instruction,
            current_image=screenshot,
            step_count=step_count,
            history_messages=[],
            history_actions=[]
        )
        
        # 调用 Agent
        start_time = time.time()
        output = agent.act(agent_input)
        elapsed = time.time() - start_time
        
        # 获取参考数据
        ref_entry = ref_data.get(current_status, [])
        if not ref_entry:
            break
        
        ref_action_list = [r.get("action", "") for r in ref_entry]
        ref_params_list = [r.get("params", {}) for r in ref_entry]
        next_status = ref_entry[0].get("next", "#") if ref_entry else "#"
        
        # 格式化显示
        pred_str = format_action(output.action, output.parameters)
        ref_str = format_action(ref_action_list[0], ref_params_list[0]) if ref_action_list else "N/A"
        
        # 比较结果
        action_match = output.action.upper() == ref_action_list[0].upper() if ref_action_list else False
        
        total_steps += 1
        if action_match:
            correct_steps += 1
            result = "✅"
        else:
            result = "❌"
        
        # 打印对比
        print(f"{step_count:<6} {current_status:<8} {pred_str:<40} {ref_str:<40} {result:<8}")
        
        # 显示 raw_output（前100字符）
        raw_preview = output.raw_output[:100].replace('\n', ' ')
        if len(output.raw_output) > 100:
            raw_preview += "..."
        print(f"       模型输出: {raw_preview}")
        print()
        
        # 更新状态
        current_status = next_status
        step_count += 1
    
    # 打印总结
    print("-" * 120)
    accuracy = correct_steps / total_steps if total_steps > 0 else 0
    print(f"总结: {correct_steps}/{total_steps} 步正确, 准确率: {accuracy:.2%}")
    print()
    
    return accuracy == 1.0, correct_steps, total_steps

def main():
    """主函数"""
    test_cases = [
        "step_meituan_onekey_0001",
        "step_douyin_onekey_0008",
        "step_baidumap_onekey_0010",
    ]
    
    print("\n" + "=" * 80)
    print("GLM-4V 详细对比测试")
    print("=" * 80)
    print()
    
    total_pass = 0
    total_cases = len(test_cases)
    total_correct_steps = 0
    total_steps = 0
    
    for case in test_cases:
        passed, correct, steps = test_case(case)
        if passed:
            total_pass += 1
        total_correct_steps += correct
        total_steps += steps
        print()
    
    # 总结果
    print("=" * 80)
    print("总结果")
    print("=" * 80)
    print(f"通过用例: {total_pass}/{total_cases}")
    print(f"总步骤: {total_correct_steps}/{total_steps}")
    print(f"总准确率: {total_correct_steps/total_steps:.2%}" if total_steps > 0 else "N/A")

if __name__ == "__main__":
    main()
