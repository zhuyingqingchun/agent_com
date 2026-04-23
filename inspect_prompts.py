#!/usr/bin/env python3
"""
Prompt 组合式审查工具

用法:
  python3 inspect_prompts.py                    # 查看第1个用例的所有步骤
  python3 inspect_prompts.py --case 2           # 查看第2个用例
  python3 inspect_prompts.py --case 1 --step 3  # 只看第1个用例的第3步
  python3 inspect_prompts.py --case 1 --step 3 --model click  # 只看click-localizer的prompt

输出格式:
  - 每个 model call 的 system + user prompt 完整展示
  - user prompt 的每段文字标注来源 (文件:行号)
  - 图片部分用 [IMAGE] 占位符表示（不输出base64）
"""

import os
import sys
import json
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import Agent
from agent_base import AgentInput, ACTION_TYPE, ACTION_CLICK


DATA_DIR = "./test_data/offline"


def load_test_cases() -> List[Dict[str, Any]]:
    cases = []
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_path}")
        sys.exit(1)

    for f in sorted(data_path.iterdir()):
        if f.is_dir():
            ref_file = f / "ref.json"
            if ref_file.exists():
                with open(ref_file, "r", encoding="utf-8") as fp:
                    ref = json.load(fp)
                overview = ref.get("case_overview", {})
                img_files = [p.name for p in f.iterdir() if p.suffix == ".png"]
                max_step = max(
                    (int(p.stem.split("-")[0]) for p in f.iterdir() if p.suffix == ".png" and p.stem.split("-")[0].isdigit()),
                    default=0,
                )
                cases.append({
                    "_dir": str(f),
                    "instruction": overview.get("instruction", ""),
                    "app": overview.get("app", ""),
                    "total_steps": max_step + 1,
                    "ref": ref,
                })
    return cases


def load_step_data(case_dir: str, step_idx: int) -> Optional[Dict[str, Any]]:
    """加载单步的截图和历史"""
    for suffix in ["", "-1"]:
        img_path = os.path.join(case_dir, f"{step_idx}{suffix}.png")
        if os.path.exists(img_path):
            break
    else:
        return None
    image = Image.open(img_path).convert("RGB")

    history = []
    ref_file = os.path.join(case_dir, "ref.json")
    if os.path.exists(ref_file):
        with open(ref_file, "r", encoding="utf-8") as fp:
            ref = json.load(fp)
        for s in range(step_idx):
            step_key = str(s)
            if step_key in ref:
                actions = ref[step_key]
                for act in actions:
                    a = act.get("action", "?")
                    p = act.get("params", {})
                    history.append({"step": s, "action": a, "parameters": p})

    return {"image": image, "history": history}


class PromptInspector(Agent):
    """包装 Agent，拦截所有 _call_api 调用，记录完整 prompt"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_calls: List[Dict[str, Any]] = []
        self._in_click_localizer = False

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """拦截 API 调用，记录 prompt 并返回 mock 响应"""
        caller = "click_localizer" if self._in_click_localizer else "main_model"
        captured = self._annotate_messages(copy.deepcopy(messages), caller=caller)
        self.captured_calls.append(captured)
        return MockResponse(action="CLICK", point=[500, 500])

    def _build_click_localizer_messages(self, *args, **kwargs):
        self._in_click_localizer = True
        result = super()._build_click_localizer_messages(*args, **kwargs)
        self._in_click_localizer = False
        return result

    def _annotate_messages(self, messages: List[Dict[str, Any]], caller: str) -> Dict[str, Any]:
        """将消息列表转换为带标注的字典"""
        annotated = {"caller": caller, "system": {}, "user": {}}
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                annotated["system"] = msg
            elif role == "user":
                annotated["user"] = msg
        return annotated


def extract_text_parts(user_content: Any) -> List[Tuple[str, str]]:
    """
    从 user content 中提取所有文本段，返回 [(text, source_tag), ...]
    这里我们通过特征字符串匹配来推断来源
    """
    parts = []

    if isinstance(user_content, str):
        return [(user_content, "raw_user_text")]

    if isinstance(user_content, list):
        for item in user_content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    tagged = tag_text_source(text)
                    parts.extend(tagged)
                elif item.get("type") == "image_url":
                    parts.append(("[IMAGE: screenshot]", "screenshot"))
                elif item.get("type") == "image_url" and "grid" in str(item.get("image_url", {}).get("url", ""))[:50]:
                    parts.append(("[IMAGE: grid overlay]", "grid_image"))

    return parts


KNOWN_TAGS = [
    ("【任务】", "agent_click_prompt.py:get_click_prompt_for_action → 任务指令"),
    ("【动作】CLICK", "agent_click_prompt.py:get_click_prompt_for_action → 动作类型"),
    ("【坐标系统", "agent_click_prompt.py:COORDINATE_SYSTEM → 坐标系说明"),
    ("【点击规则】", "agent_click_prompt.py:get_click_guidance → 点击规则"),
    ("【当前点击目标】", "agent_click_prompt.py:get_click_guidance → target_element"),
    ("【方向参考", "agent_click_prompt.py:DIRECTION_HINTS → 阶段方向提示(phase)"),
    ("【输出要求】", "agent_click_prompt.py:get_click_prompt_for_action → 输出格式"),
    ("【当前子目标】", "agent.py:_build_click_localizer_messages → get_current_subgoal()"),
    ("【上一步操作】TYPE", "agent.py:L414 → op_context (last_action=TYPE)"),
    ("【上一步操作】CLICK", "agent.py:L420 → op_context (last_action=CLICK+top)"),
    ("【当前目的】需要点击", "agent.py:L415/L421 → op_context 目的"),
    ("【目标特征】确认类按钮", "agent.py:L416 → op_context 特征"),
    ("⚠️ 绝对不要点左上角", "agent.py:L417 → op_context 禁止行为"),
    ("⚠️ 如果主模型判断本步是CLICK", "agent.py:L422 → op_context 重复点击警告"),
    ("【主模型判断的大致区域】", "agent.py:L437 → region_hint (REGION_SEMANTIC)"),
    ("【主模型粗略方位参考】", "agent.py:L444 → coarse_point area_desc"),
    ("【阶段方向提示】", "agent.py:L455 → DIRECTION_HINTS (phase)"),
    ("仅做点击定位", "agent.py:L457 → 尾部固定文字"),
    ("任务:", "agent_prompt.py:get_user_prompt → instruction"),
    ("应用:", "agent_prompt.py:get_user_prompt → app_name"),
    ("已输入:", "agent_prompt.py:get_user_prompt → typed_texts"),
    ("输出JSON:", "agent_prompt.py:get_user_prompt → 尾部"),
    ("你是安卓 UI 自动化助手", "agent_prompt.py:OptimizedPrompt.get_system_prompt → system"),
    ("【动作格式】", "agent_prompt.py:OptimizedPrompt → 动作格式定义"),
    ("【关键词提取】", "agent_prompt.py:OptimizedPrompt → 关键词提取"),
    ("【标准流程】", "agent_prompt.py:OptimizedPrompt → 标准流程"),
    ("【工作流步骤】", "agent_prompt.py:_build_progress_table → workflow_steps"),
    ("【历史动作】", "agent_prompt.py:_build_progress_table → history"),
    ("【候选区域记忆】", "agent_prompt.py:_build_progress_table → region_memory"),
    ("【Playbook信息】", "agent_prompt.py:_build_progress_table → playbook_info"),
    ("你是GUI点击坐标精确定位器", "agent.py:L461 → click-localizer system prompt"),
    ("【坐标规则】", "agent.py:L463 → click-localizer system 坐标规则"),
    ("【命中优先级】", "agent.py:L469 → click-localizer system 命中优先级"),
    ("【禁止行为】", "agent.py:L474 → click-localizer system 禁止行为"),
]


def tag_text_source(text: str) -> List[Tuple[str, str]]:
    """根据文本内容特征推断来源"""
    if not text.strip():
        return []

    lines = text.split("\n")
    result = []
    current_block: List[str] = []
    current_source = "unknown"

    for line in lines:
        matched = False
        for prefix, source in KNOWN_TAGS:
            if line.strip().startswith(prefix[:6]) or line.strip().startswith(prefix[:4]):
                if current_block:
                    result.append(("\n".join(current_block).strip(), current_source))
                    current_block = []
                current_block.append(line)
                current_source = source
                matched = True
                break
        if not matched:
            current_block.append(line)

    if current_block:
        result.append(("\n".join(current_block).strip(), current_source))

    if not result and text.strip():
        result = [(text.strip(), "unmatched_text")]

    return result


class MockResponse:
    def __init__(self, action="CLICK", point=[500, 500]):
        self.choices = [MockChoice(action, point)]


class MockChoice:
    def __init__(self, action, point):
        self.message = MockMessage(action, point)


class MockMessage:
    def __init__(self, action, point):
        payload = {"action": action, "parameters": {"point": point}}
        self.content = json.dumps(payload, ensure_ascii=False)


def render_call(call: Dict[str, Any], index: int, filter_model: Optional[str] = None) -> str:
    """渲染一次 API 调用的完整 prompt"""
    caller = call.get("caller", "?")
    if filter_model and caller != filter_model:
        return ""

    lines = []
    sep = "=" * 80
    lines.append(f"\n{'━' * 80}")
    lines.append(f"  📞 Call #{index}  [{caller.upper()}]")
    lines.append(f"{'━' * 80}")

    # System prompt
    sys_msg = call.get("system", {})
    sys_text = ""
    if isinstance(sys_msg.get("content"), str):
        sys_text = sys_msg["content"]
    elif isinstance(sys_msg.get("content"), list):
        for item in sys_msg["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                sys_text += item.get("text", "") + "\n"

    if sys_text.strip():
        lines.append(f"\n  📋 SYSTEM PROMPT ({len(sys_text)} chars):")
        lines.append(f"  {'─' * 60}")
        sys_tagged = tag_text_source(sys_text)
        for block_text, source in sys_tagged:
            lines.append(f"  │ 📌 {source}")
            for bl in block_text.split("\n"):
                lines.append(f"  │   {bl}")
        lines.append(f"  {'─' * 60}")

    # User prompt
    user_msg = call.get("user", {})
    user_content = user_msg.get("content", "")

    lines.append(f"\n  👤 USER PROMPT:")
    lines.append(f"  {'─' * 60}")

    parts = extract_text_parts(user_content)
    for part_text, source in parts:
        if part_text.startswith("[IMAGE:"):
            lines.append(f"  🖼️  {part_text}")
        else:
            tagged_blocks = tag_text_source(part_text)
            for block_text, block_source in tagged_blocks:
                lines.append(f"  │ 📌 {block_source}")
                for bl in block_text.split("\n"):
                    lines.append(f"  │   {bl}")
                lines.append("")

    lines.append(f"  {'─' * 60}")

    # Summary stats
    total_chars = len(sys_text) + len(str(user_content)) if isinstance(user_content, str) else len(sys_text) + sum(
        len(p[0]) for p in parts if not p[0].startswith("[IMAGE:")
    )
    image_count = sum(1 for p in parts if "[IMAGE:" in p[0])
    lines.append(f"\n  📊 总字符数: ~{total_chars} | 图片数: {image_count}")

    return "\n".join(lines)


def inspect_case(case_index: int, step_filter: Optional[int] = None, model_filter: Optional[str] = None):
    """审查单个测试用例的所有步骤"""
    cases = load_test_cases()
    if case_index < 1 or case_index > len(cases):
        print(f"❌ 用例范围 1-{len(cases)}，你选了 {case_index}")
        return

    case = cases[case_index - 1]
    case_dir = case["_dir"]
    instruction = case.get("instruction", "(无指令)")
    total_steps = case.get("total_steps", "?")

    print(f"\n{'█' * 80}")
    print(f"  📦 用例 #{case_index}: {instruction[:60]}...")
    print(f"  📁 目录: {case_dir}")
    print(f"  📏 总步数: {total_steps}")
    print(f"{'█' * 80}")

    import logging
    logging.disable(logging.CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")

    inspector = PromptInspector()

    max_step = step_filter or total_steps
    call_counter = 0

    for step in range(1, max_step + 1):
        step_data = load_step_data(case_dir, step)
        if step_data is None:
            print(f"\n  ⚠️ Step {step}: 无截图数据，跳过")
            continue

        input_data = AgentInput(
            instruction=instruction,
            current_image=step_data["image"],
            step_count=step,
            history_actions=step_data["history"],
        )

        pre_calls = len(inspector.captured_calls)

        try:
            output = inspector.act(input_data)
        except Exception as e:
            print(f"\n  ❌ Step {step} 执行异常: {e}")
            continue

        new_calls = inspector.captured_calls[pre_calls:]

        if step_filter:
            for i, call in enumerate(new_calls):
                call_counter += 1
                rendered = render_call(call, call_counter, filter_model=model_filter)
                if rendered:
                    print(rendered)
        else:
            print(f"\n  ▶ Step {step}: action={output.action}, params={output.parameters}  ({len(new_calls)} calls)")
            if model_filter is None:
                print(f"     (用 --step {step} 查看详细 prompt)")

    if not step_filter:
        print(f"\n\n💡 提示: 用 --step N 查看某步的完整 prompt 内容")


def main():
    parser = argparse.ArgumentParser(description="Prompt 组合式审查工具")
    parser.add_argument("--case", type=int, default=1, help="用例编号 (从1开始)")
    parser.add_argument("--step", type=int, default=None, help="只查看指定步骤")
    parser.add_argument("--model", type=str, default=None, choices=["main", "click"],
                        help="过滤: main=主模型, click=click-localizer")
    args = parser.parse_args()

    inspect_case(args.case, args.step, args.model)


if __name__ == "__main__":
    main()
