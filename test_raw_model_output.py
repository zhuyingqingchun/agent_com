#!/usr/bin/env python3
"""测试并显示模型的原始输出"""

import os
import sys
sys.path.insert(0, './submission/src')

# 设置环境变量
os.environ['VLM_API_KEY'] = '60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
os.environ['DEBUG_API_URL'] = 'https://open.bigmodel.cn/api/paas/v4'
os.environ['DEBUG_MODEL_ID'] = 'glm-4v-flash'

from pathlib import Path
from PIL import Image
from agent import Agent
from agent_base import AgentInput

# 初始化 Agent
agent = Agent()

# 选择测试用例
test_case = "step_meituan_onekey_0001"
test_dir = Path(f"./test_data/offline/{test_case}")

print("=" * 60)
print(f"测试用例: {test_case}")
print(f"模型: glm-4v-flash")
print("=" * 60)
print()

# 读取第一张截图
screenshot_files = sorted(test_dir.glob("*.png"))
if not screenshot_files:
    print(f"错误: 未找到截图文件")
    sys.exit(1)

first_screenshot = Image.open(screenshot_files[0])

# 从 ref.json 读取指令
import json
ref_file = test_dir / "ref.json"
with open(ref_file, 'r') as f:
    ref_data = json.load(f)

instruction = ref_data.get("case_overview", {}).get("instruction", "")
print(f"指令: {instruction}")
print()

# 创建 AgentInput
agent_input = AgentInput(
    instruction=instruction,
    current_image=first_screenshot,
    step_count=1,
    history_messages=[],
    history_actions=[]
)

# 调用 Agent
print("正在调用模型...")
print("-" * 60)
output = agent.act(agent_input)

print(f"动作: {output.action}")
print(f"参数: {output.parameters}")
print()
print("=" * 60)
print("模型原始输出 (raw_output):")
print("=" * 60)
print(output.raw_output)
print()

if output.usage:
    print("-" * 60)
    print(f"Token 使用: 输入={output.usage.get('prompt_tokens', 'N/A')}, 输出={output.usage.get('completion_tokens', 'N/A')}")
