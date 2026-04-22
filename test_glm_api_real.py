#!/usr/bin/env python3
"""测试真实 GLM-4V API 调用"""

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
import time

# 初始化 Agent
agent = Agent()

# 选择测试用例
test_case = "step_meituan_onekey_0001"
test_dir = Path(f"./test_data/offline/{test_case}")

print("=" * 60)
print(f"真实 GLM-4V API 测试")
print(f"测试用例: {test_case}")
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
print("正在调用 GLM-4V API...")
print("-" * 60)
start_time = time.time()
output = agent.act(agent_input)
end_time = time.time()

print(f"调用耗时: {end_time - start_time:.2f} 秒")
print(f"动作: {output.action}")
print(f"参数: {output.parameters}")
print()
print("=" * 60)
print("模型原始输出 (raw_output):")
print("=" * 60)
print(output.raw_output[:500] if len(output.raw_output) > 500 else output.raw_output)
print()

# 检查是否包含 fallback 或 api 字样
if "fallback" in output.raw_output.lower():
    print("⚠️ 警告: 检测到 fallback，可能是 API 调用失败")
elif "offline_match" in output.raw_output.lower():
    print("⚠️ 警告: 检测到 offline_match，离线匹配未禁用")
else:
    print("✅ 成功调用 GLM-4V API")

if output.usage:
    print(f"\nToken 使用:")
    print(f"  输入: {output.usage.prompt_tokens}")
    print(f"  输出: {output.usage.completion_tokens}")
