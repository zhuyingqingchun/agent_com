#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用仓库官方 TestRunner 进行本地测试，但把模型切到自定义 OpenAI 兼容服务，
并记录每一次模型调用的 prompt、模型输出、步骤号、阶段、来源文件到 output/prompt_trace.jsonl。

增强版特性：
1. 支持全量 / 单用例测试
2. 为每次模型请求设置超时，避免“卡死”
3. 每次请求前后写独立请求日志 output/api_call_trace.jsonl
4. 可选关闭 extra_body.thinking，避免某些 OpenAI 兼容服务不支持
5. 出错时把最后一次调用上下文写入日志，便于定位是服务端、模型还是 prompt 问题
"""

import os
import json
import time
import argparse
import inspect
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI


# -----------------------------
# 1) 自定义 OpenAI 兼容服务配置
# -----------------------------
CUSTOM_BASE_URL = "http://10.100.26.189:8000/v1"
CUSTOM_API_KEY = "not-needed"
CUSTOM_MODEL_ID = "/mnt/PRO6000_disk/models/Qwen/Qwen2___5-VL-72B-Instruct"


# -----------------------------
# 2) CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run official TestRunner with prompt tracing and optional single-case mode."
    )
    parser.add_argument("--data-dir", type=str, default="./test_data/offline",
                        help="官方离线测试数据目录；全量测试时扫描其下所有 case。")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="输出目录。")
    parser.add_argument("--debug-test", type=str, default="true",
                        help="是否 debug_test 模式（true/false）。true=错误后继续跑，false=错误后终止。")
    parser.add_argument("--case-dir", type=str, default="",
                        help="单用例测试：直接指定某个 case 目录（目录内需包含 ref.json）。")
    parser.add_argument("--case-name", type=str, default="",
                        help="单用例测试：指定相对于 data_dir 的 case 名。")
    parser.add_argument("--request-timeout", type=float, default=90.0,
                        help="单次模型请求超时秒数，默认 90 秒。")
    parser.add_argument("--disable-thinking-extra-body", action="store_true",
                        help="不发送 extra_body={'thinking': {'type': 'disabled'}}，兼容部分 OpenAI 服务。")
    parser.add_argument("--max-completion-tokens", type=int, default=1024,
                        help="每次模型调用的 max_tokens，默认 1024。")
    return parser.parse_args()


# -----------------------------
# 3) prompt 来源映射
# -----------------------------
TEMPLATE_CLASS_MAP = {
    "optimized": "OptimizedPrompt",
    "action": "ActionPrompt",
    "grounded_action": "GroundedActionPrompt",
    "task_guided": "TaskGuidedPrompt",
    "visual_grounding": "VisualGroundingPrompt",
    "simple": "SimplePrompt",
    "detailed": "DetailedPrompt",
}

PROMPT_STAGE_HINTS = {
    ("agent.py", "act"): "main_action",
    ("agent.py", "_localize_click"): "click_localizer",
    ("agent.py", "_plan_workflow"): "workflow_planner",
    ("agent_actions.py", "_call_phase_corrector"): "phase_corrector",
    ("agent_actions.py", "_call_left_panel_corrector"): "left_panel_corrector",
    ("agent_actions.py", "_call_completion_checker"): "completion_checker",
}


def resolve_prompt_origin(agent_obj: Any, caller_file: str, caller_func: str) -> Dict[str, Any]:
    stage = PROMPT_STAGE_HINTS.get((caller_file, caller_func), "unknown")

    template_name = getattr(getattr(agent_obj, "features", None), "prompt_template", "unknown")
    template_class = TEMPLATE_CLASS_MAP.get(template_name, template_name)

    if (caller_file, caller_func) == ("agent.py", "act"):
        return {
            "stage": stage,
            "prompt_source_files": [
                f"utils/agent_prompt.py::{template_class}.get_system_prompt",
                f"utils/agent_prompt.py::{template_class}.get_user_prompt",
                "agent.py::Agent._build_messages",
            ],
            "notes": "主动作决策 prompt。当前仓库默认 features.prompt_template = grounded_action。",
        }

    if (caller_file, caller_func) == ("agent.py", "_localize_click"):
        return {
            "stage": stage,
            "prompt_source_files": [
                "utils/agent_click_prompt.py::ClickPromptHelper.get_click_prompt_for_action",
                "agent.py::Agent._build_click_localizer_messages",
            ],
            "notes": "CLICK 二阶段精定位 prompt。",
        }

    if (caller_file, caller_func) == ("agent.py", "_plan_workflow"):
        return {
            "stage": stage,
            "prompt_source_files": [
                "agent.py::Agent._plan_workflow",
            ],
            "notes": "多步任务分解 prompt，直接写在 agent.py 里。",
        }

    if (caller_file, caller_func) == ("agent_actions.py", "_call_phase_corrector"):
        return {
            "stage": stage,
            "prompt_source_files": [
                "utils/agent_actions.py::ActionProcessor._call_phase_corrector",
            ],
            "notes": "detail/confirm 阶段顶部误点纠偏 prompt。",
        }

    if (caller_file, caller_func) == ("agent_actions.py", "_call_left_panel_corrector"):
        return {
            "stage": stage,
            "prompt_source_files": [
                "utils/agent_actions.py::ActionProcessor._call_left_panel_corrector",
            ],
            "notes": "detail 阶段左侧栏误点纠偏 prompt。",
        }

    if (caller_file, caller_func) == ("agent_actions.py", "_call_completion_checker"):
        return {
            "stage": stage,
            "prompt_source_files": [
                "utils/agent_actions.py::ActionProcessor._call_completion_checker",
            ],
            "notes": "confirm 阶段过早 COMPLETE 纠偏 prompt。",
        }

    return {
        "stage": stage,
        "prompt_source_files": [f"{caller_file}::{caller_func}"],
        "notes": "未命中已知映射，按调用栈记录。",
    }


# -----------------------------
# 4) 日志工具
# -----------------------------
def sanitize_image_url(url: str) -> str:
    if not isinstance(url, str):
        return "<non-string-image-url>"
    if url.startswith("data:image"):
        prefix, _, payload = url.partition(",")
        return f"{prefix},<base64_omitted:{len(payload)}chars>"
    if len(url) > 200:
        return url[:200] + "...<truncated>"
    return url


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            safe.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            items = []
            for item in content:
                if not isinstance(item, dict):
                    items.append({"type": "unknown", "value": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")
                    items.append({"type": "image_url", "image_url": {"url": sanitize_image_url(url)}})
                else:
                    items.append({"type": item_type or "unknown", "value": str(item)})
            safe.append({"role": role, "content": items})
            continue

        safe.append({"role": role, "content": str(content)})
    return safe


def extract_text_blocks(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if isinstance(content, str):
            blocks.append({"role": role, "text": content})
            continue

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"role": role, "text": item.get("text", "")})
    return blocks


def serialize_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result: Dict[str, Any] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }

    prompt_details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None) or getattr(usage, "output_tokens_details", None)

    if prompt_details is not None:
        result["prompt_tokens_details"] = {
            "cached_tokens": getattr(prompt_details, "cached_tokens", None)
        }
    if completion_details is not None:
        result["completion_tokens_details"] = {
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None)
        }
    return result


def safe_get_model_output(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def find_prompt_caller() -> Tuple[str, str, Dict[str, Any]]:
    stack = inspect.stack()
    try:
        for frame_info in stack[1:]:
            filename = frame_info.filename.replace("\\", "/")
            base = os.path.basename(filename)
            if base in {"agent.py", "agent_actions.py"}:
                return base, frame_info.function, dict(frame_info.frame.f_locals)
    finally:
        del stack
    return "unknown", "unknown", {}


def extract_step_and_instruction(frame_locals: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    input_data = frame_locals.get("input_data")
    if input_data is not None:
        return getattr(input_data, "step_count", None), getattr(input_data, "instruction", None)
    return None, None


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_prompt_source_map(output_dir: Path):
    md = output_dir / "prompt_source_map.md"
    text = """# Prompt Source Map

这个文件说明 `prompt_trace.jsonl` 里不同阶段的 prompt 来自哪里。

## 1. 主动作决策（main_action）
- `utils/agent_prompt.py::{具体模板类}.get_system_prompt`
- `utils/agent_prompt.py::{具体模板类}.get_user_prompt`
- `agent.py::Agent._build_messages`

当前仓库 `Agent.__init__()` 里默认 `prompt_template="grounded_action"`。

## 2. 点击二阶段定位（click_localizer）
- `utils/agent_click_prompt.py::ClickPromptHelper.get_click_prompt_for_action`
- `agent.py::Agent._build_click_localizer_messages`

## 3. 工作流规划（workflow_planner）
- `agent.py::Agent._plan_workflow`

## 4. 顶部误点纠偏（phase_corrector）
- `utils/agent_actions.py::ActionProcessor._call_phase_corrector`

## 5. 左侧栏误点纠偏（left_panel_corrector）
- `utils/agent_actions.py::ActionProcessor._call_left_panel_corrector`

## 6. 过早 COMPLETE 纠偏（completion_checker）
- `utils/agent_actions.py::ActionProcessor._call_completion_checker`
"""
    md.write_text(text, encoding="utf-8")


def write_run_notes(output_dir: Path):
    md = output_dir / "README_trace.md"
    text = """# 运行输出说明

## 关键文件

- `prompt_trace.jsonl`
  - 每次模型调用一条记录
  - 包含：调用来源、阶段、step、prompt 文本、脱敏后的 messages、模型输出、usage

- `api_call_trace.jsonl`
  - 每次真实 HTTP 推理请求一条记录
  - 包含：开始时间、结束时间、耗时、是否成功、异常信息
  - 如果卡死问题解决后，这个文件最适合看“到底卡在哪一步”

- `single_case_result.json`
  - 单用例模式下的汇总结果

- `run_summary.json`
  - 全量模式下 TestRunner 的汇总结果

- `test_run.log`
  - 官方 TestRunner 日志

## 推荐查看顺序

1. 先看 `api_call_trace.jsonl`
   - 是否已经发出请求
   - 请求是否超时
   - 耗时多少
2. 再看 `prompt_trace.jsonl`
   - prompt 文本是什么
   - 模型输出是什么
   - 由哪个文件构造
3. 最后结合 `test_run.log` 看 checker 失败在哪一步
"""
    md.write_text(text, encoding="utf-8")


def build_openai_client(agent_obj: Any, request_timeout: float) -> OpenAI:
    return OpenAI(
        base_url=agent_obj.api_url,
        api_key=agent_obj.api_key,
        timeout=request_timeout,
        max_retries=0,
    )


def custom_call_api(agent_obj: Any,
                    messages: List[Dict[str, Any]],
                    *,
                    request_timeout: float,
                    disable_thinking_extra_body: bool,
                    max_completion_tokens: int):
    client = build_openai_client(agent_obj, request_timeout=request_timeout)

    create_kwargs: Dict[str, Any] = {
        "model": agent_obj.model_id,
        "messages": messages,
        "max_tokens": max_completion_tokens,
    }

    if not disable_thinking_extra_body:
        create_kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled"
            }
        }

    return client.chat.completions.create(**create_kwargs)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "prompt_trace.jsonl"
    api_trace_path = output_dir / "api_call_trace.jsonl"
    for p in [trace_path, api_trace_path]:
        if p.exists():
            p.unlink()

    os.environ["DEBUG_API_URL"] = CUSTOM_BASE_URL
    os.environ["DEBUG_MODEL_ID"] = CUSTOM_MODEL_ID
    os.environ["VLM_API_KEY"] = CUSTOM_API_KEY
    if "EVAL_MODE" in os.environ:
        del os.environ["EVAL_MODE"]

    write_prompt_source_map(output_dir)
    write_run_notes(output_dir)

    from agent import Agent
    from test_runner import TestRunner
    from agent_base import BaseAgent

    original_call_api = BaseAgent._call_api
    call_counter = {"value": 0}

    def traced_call_api(self, messages, **kwargs):
        call_counter["value"] += 1
        call_id = call_counter["value"]

        caller_file, caller_func, frame_locals = find_prompt_caller()
        step_count, instruction = extract_step_and_instruction(frame_locals)
        origin = resolve_prompt_origin(self, caller_file, caller_func)

        common_context = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_config": {
                "base_url": self.api_url,
                "model_id": self.model_id,
                "api_key": "<hidden>" if self.api_key else "",
                "request_timeout": args.request_timeout,
                "max_tokens": args.max_completion_tokens,
                "disable_thinking_extra_body": args.disable_thinking_extra_body,
            },
            "caller": {
                "file": caller_file,
                "function": caller_func,
            },
            "prompt_origin": origin,
            "task_context": {
                "step_count": step_count,
                "instruction": instruction,
                "phase": getattr(self, "_state", {}).get("phase") if hasattr(self, "_state") else None,
                "app_name": getattr(self, "_state", {}).get("app_name") if hasattr(self, "_state") else None,
                "task_type": getattr(self, "_state", {}).get("task_type") if hasattr(self, "_state") else None,
            },
            "messages_text_only": extract_text_blocks(messages),
            "messages_sanitized": sanitize_messages(messages),
        }

        api_start = time.time()
        write_jsonl(api_trace_path, {
            "call_id": call_id,
            "event": "request_start",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "caller": common_context["caller"],
            "task_context": common_context["task_context"],
            "request_timeout": args.request_timeout,
        })

        try:
            response = custom_call_api(
                self,
                messages,
                request_timeout=args.request_timeout,
                disable_thinking_extra_body=args.disable_thinking_extra_body,
                max_completion_tokens=args.max_completion_tokens,
            )
            elapsed = round(time.time() - api_start, 3)

            prompt_record = dict(common_context)
            prompt_record["ok"] = True
            prompt_record["elapsed_seconds"] = elapsed
            prompt_record["model_output"] = safe_get_model_output(response)
            prompt_record["usage"] = serialize_usage(response)
            write_jsonl(trace_path, prompt_record)

            write_jsonl(api_trace_path, {
                "call_id": call_id,
                "event": "request_success",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "caller": common_context["caller"],
                "task_context": common_context["task_context"],
                "usage": serialize_usage(response),
            })
            return response

        except Exception as exc:
            elapsed = round(time.time() - api_start, 3)

            prompt_record = dict(common_context)
            prompt_record["ok"] = False
            prompt_record["elapsed_seconds"] = elapsed
            prompt_record["error"] = repr(exc)
            write_jsonl(trace_path, prompt_record)

            write_jsonl(api_trace_path, {
                "call_id": call_id,
                "event": "request_error",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "caller": common_context["caller"],
                "task_context": common_context["task_context"],
                "error": repr(exc),
            })
            raise

    BaseAgent._call_api = traced_call_api

    try:
        agent = Agent()
        debug_test = str(args.debug_test).lower() == "true"
        runner = TestRunner(agent=agent, debug_test=debug_test)

        case_dir = str(args.case_dir).strip()
        case_name = str(args.case_name).strip()
        data_dir = Path(args.data_dir).resolve()

        if case_dir and case_name:
            raise ValueError("只能指定 --case-dir 或 --case-name 其中一个，不能同时指定。")

        if case_dir or case_name:
            if case_dir:
                target_case_dir = Path(case_dir).resolve()
            else:
                target_case_dir = (data_dir / case_name).resolve()

            ref_json = target_case_dir / "ref.json"
            if not ref_json.exists():
                raise FileNotFoundError(f"单用例目录不存在 ref.json: {ref_json}")

            case_name_final = target_case_dir.name
            case_output_dir = output_dir / case_name_final
            case_output_dir.mkdir(parents=True, exist_ok=True)

            result = runner.run_task(
                screenshots_dir=str(target_case_dir),
                visualization_dir=str(case_output_dir / "visualization"),
            )

            single_case_result = {
                "mode": "single_case",
                "case_dir": str(target_case_dir),
                "case_name": case_name_final,
                "instruction": result.get("instruction"),
                "current_status": result.get("current_status"),
                "next_status": result.get("next_status"),
                "visualization_path": result.get("visualization_path"),
                "step_count": len(result.get("steps", [])),
                "pass_step_count": sum(1 for step in result.get("steps", []) if step.get("check_result")),
                "steps": result.get("steps", []),
                "trace_file": str(trace_path),
                "api_trace_file": str(api_trace_path),
            }

            summary_path = output_dir / "single_case_result.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(single_case_result, f, ensure_ascii=False, indent=2)

            print("=" * 80)
            print("单用例测试完成")
            print(f"用例目录: {target_case_dir}")
            print(f"结果汇总: {summary_path}")
            print(f"Prompt + 模型输出跟踪: {trace_path}")
            print(f"API 调用跟踪: {api_trace_path}")
            print("=" * 80)

        else:
            summary = runner.run_all_tasks(
                data_dir=str(data_dir),
                output_dir=str(output_dir),
            )

            summary_path = output_dir / "run_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            print("=" * 80)
            print("全量测试完成")
            print(f"官方日志: {output_dir / 'test_run.log'}")
            print(f"结果汇总: {summary_path}")
            print(f"Prompt + 模型输出跟踪: {trace_path}")
            print(f"API 调用跟踪: {api_trace_path}")
            print("=" * 80)

    finally:
        BaseAgent._call_api = original_call_api


if __name__ == "__main__":
    main()
