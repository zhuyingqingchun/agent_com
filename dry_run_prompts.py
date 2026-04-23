"""Dry-run: 打印美团测试每步 prompt（含 Playbook 按需加载流程）。"""
import json, sys, os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.agent_state import make_initial_state, reset_task_state
from utils.agent_rules import extract_app_name, infer_task_type
from utils.agent_prompt import GroundedActionPrompt
from utils.task_playbook import classify_task, TaskPlaybook


def load_meituan_data():
    base = "test_data/offline/step_meituan_onekey_0001"
    with open(f"{base}/ref.json") as f:
        ref = json.load(f)
    instruction = ref["case_overview"]["instruction"]
    return ref, instruction


def simulate_steps(ref, instruction):
    prompt_template = GroundedActionPrompt()
    
    print("=" * 80)
    print(f"指令: {instruction}")
    
    pb_info = classify_task(instruction, "美团")
    if pb_info:
        print(f"✅ Playbook 匹配: {pb_info['name']} ({pb_info['type']})")
        print(f"   流程共 {len(pb_info['flow'])} 步")
    else:
        print("⚪ 未匹配到任何 Playbook（不加载流程）")
    
    status_order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "14"]
    
    for step_idx, status in enumerate(status_order):
        step_num = step_idx + 1
        
        history = []
        for i, prev_status in enumerate(status_order[:step_idx]):
            pd = ref.get(prev_status, [{}])[0]
            history.append({"step": i + 1, "action": pd.get("action", "?"), "parameters": pd.get("params", {})})
        
        user_text = prompt_template.get_user_prompt(
            instruction=instruction,
            state={"typed_texts": ["窑村干锅猪蹄"] if step_num >= 5 else [],
                   "app_name": "美团"},
            history=history,
            workflow_hint="",
            app_memory="",
            current_subgoal="",
            workflow_steps=[],
            playbook_info=pb_info,
        )
        
        step_data = ref.get(status, [{}])[0]
        expected_action = step_data.get("action", "?")
        
        if step_num in [1, 2, 3, 7, 8, 9]:
            print(f"\n{'='*80}")
            print(f"Step {step_num} | 状态={status} | 期望={expected_action}")
            print(f"{'─'*80}")
            
            lines = user_text.split("\n")
            in_progress = False
            for line in lines:
                if "已完成操作" in line or "标准流程" in line or "当前判断" in line or "提示" in line:
                    in_progress = True
                if in_progress:
                    print(line)
                if in_progress and line.strip() == "":
                    if "当前判断" in "\n".join(lines[max(0,lines.index(line)-3):]):
                        break
    
    print(f"\n{'='*80}")
    print("完整 Prompt 示例 (Step 8):")
    print(f"{'─'*80}")


if __name__ == "__main__":
    ref, inst = load_meituan_data()
    simulate_steps(ref, inst)
