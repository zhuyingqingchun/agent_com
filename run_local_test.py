#!/usr/bin/env python3
"""
本地测试启动脚本

功能：
1. 配置本地 Qwen 模型 API（通过环境变量）
2. 启用 prompt 记录（每步完整 prompt + 来源标注，写入 ./prompt_logs/）
3. 调用官方 test_runner.py 运行测试

用法:
  python3 run_local_test.py                        # 跑全部用例
  python3 run_local_test.py --case 8               # 只跑美团
  python3 run_local_test.py --case 8 --step-limit 5 # 美团最多跑5步

输出:
  ./prompt_logs/prompts.json     ← 所有步骤的完整 prompt 记录（含来源标注）
  ./output/                      ← 官方测试结果
"""

import os
import sys
import json
import argparse
from datetime import datetime


# ==========================================
#  本地模型配置（环境变量方式传入 agent_base.py）
# ==========================================
os.environ["DEBUG_API_URL"] = "http://10.100.26.189:8000/v1"
os.environ["DEBUG_MODEL_ID"] = "/mnt/PRO6000_disk/models/Qwen/Qwen2___5-VL-72B-Instruct"
os.environ["VLM_API_KEY"] = "not-needed"

LOG_DIR = "./prompt_logs"


def main():
    parser = argparse.ArgumentParser(description="本地测试 + Prompt 记录")
    parser.add_argument("--case", type=int, default=None, help="只跑指定编号的用例 (1-based)")
    parser.add_argument("--step-limit", type=int, default=None, help="限制最大步数（用于快速验证）")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Prompt 日志目录")
    args, remaining = parser.parse_known_args()

    os.makedirs(args.log_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  本地测试配置:")
    print(f"    API URL:    {os.environ['DEBUG_API_URL']}")
    print(f"    Model ID:   {os.environ['DEBUG_MODEL_ID']}")
    print(f"    API Key:    {os.environ['VLM_API_KEY']}")
    print(f"    Prompt Log: {args.log_dir}/prompts.json")
    print(f"{'='*60}")

    from agent import Agent

    original_init = Agent.__init__

    def patched_init(self, *a, **kw):
        original_init(self, *a, **kw)
        self.enable_prompt_logging(args.log_dir)
        if args.step_limit is not None:
            self._step_limit = args.step_limit

    original_act = Agent.act

    def patched_act(self, input_data):
        if hasattr(self, '_step_limit') and input_data.step_count > self._step_limit:
            from agent_base import ACTION_COMPLETE, AgentOutput
            print(f"\n  ⏹ Step {input_data.step_count} 达到步数限制 ({self._step_limit})，强制 COMPLETE")
            return AgentOutput(action=ACTION_COMPLETE, parameters={})
        return original_act(self, input_data)

    Agent.__init__ = patched_init
    Agent.act = patched_act

    sys.argv = ["test_runner.py"] + remaining
    from test_runner import TestRunner

    agent = Agent()
    runner = TestRunner(agent, debug_test=True)

    results = runner.run_all_tasks()
    
    print(f"\n{'='*60}")
    print(f"  Prompt 日志已保存到: {args.log_dir}/prompts.json")
    print(f"  总共记录了 {len(agent._prompt_calls)} 次 API 调用的 prompt")
    print(f"{'='*60}")
    
    _print_prompt_summary(agent._prompt_calls)


def _print_prompt_summary(calls):
    """打印 prompt 记录摘要"""
    if not calls:
        print("  （无记录）")
        return
    
    print(f"\n  📋 Prompt 记录摘要:")
    print(f"  {'─'*60}")
    for call in calls:
        idx = call["index"]
        caller = call["caller"].upper()
        ts = call.get("timestamp", "")[11:19]
        
        sys_text = call.get("system", "")
        user_text = call.get("user", "")
        sys_parts = call.get("system_parts", [])
        user_parts = call.get("user_parts", [])
        
        total_chars = len(sys_text) + len(user_text)
        img_count = user_text.count("[IMAGE]") if user_text else 0
        
        print(f"  #{idx:3d} [{caller:16s}] {ts} | ~{total_chars:5d} chars | {img_count} images")

        for part in user_parts[:5]:
            src = part.get("source", "?")
            text_preview = part.get("text", "")[:50].replace("\n", " ")
            print(f"       ├─ 📌 {src}")
            print(f"       │  {text_preview}...")
        if len(user_parts) > 5:
            print(f"       └─ ... 还有 {len(user_parts)-5} 段")
    
    print(f"\n  💡 查看完整内容: cat {LOG_DIR}/prompts.json | python3 -m json.tool")


if __name__ == "__main__":
    main()
