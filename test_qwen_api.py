#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试本地 OpenAI 兼容 API 的脚本
支持：
1. 纯文本对话
2. 文本 + 图片多模态
3. 超时控制
4. 打印耗时、usage、模型输出
5. 可选保存完整响应到 json

示例：

# 纯文本
python test_qwen_api.py \
  --prompt "你好，介绍一下你自己" \
  --timeout 60

# 图片 + 文本
python test_qwen_api.py \
  --image ./demo.jpg \
  --prompt "描述这张图片" \
  --timeout 60

# 保存结果
python test_qwen_api.py \
  --image ./demo.jpg \
  --prompt "描述这张图片" \
  --save-json ./api_result.json
"""

import io
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI
from PIL import Image


DEFAULT_BASE_URL = "http://10.100.26.189:8000/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_MODEL_ID = "/mnt/PRO6000_disk/models/Qwen/Qwen2___5-VL-72B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Test local OpenAI-compatible API.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI 兼容服务地址")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="模型 ID")
    parser.add_argument("--prompt", type=str, required=True, help="输入给模型的文本 prompt")
    parser.add_argument("--image", type=str, default="", help="可选图片路径")
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大输出 token")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--save-json", type=str, default="", help="可选：保存结果到 json 文件")
    parser.add_argument(
        "--disable-thinking-extra-body",
        action="store_true",
        help="不发送 extra_body={'thinking': {'type': 'disabled'}}，兼容部分服务",
    )
    return parser.parse_args()


def encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
        fmt = "JPEG"
    elif suffix == ".png":
        mime = "image/png"
        fmt = "PNG"
    elif suffix == ".webp":
        mime = "image/webp"
        fmt = "WEBP"
    else:
        # 兜底：按 PNG 转
        mime = "image/png"
        fmt = "PNG"

    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{img_base64}"


def response_to_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "content": response.choices[0].message.content if getattr(response, "choices", None) else None,
        "finish_reason": response.choices[0].finish_reason if getattr(response, "choices", None) else None,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        },
    }


def main():
    args = parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    if args.image:
        image_url = encode_image_to_data_url(args.image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": args.prompt,
            }
        ]

    create_kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    if not args.disable_thinking_extra_body:
        create_kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled"
            }
        }

    print("=" * 80)
    print("开始请求")
    print(f"base_url   : {args.base_url}")
    print(f"model      : {args.model}")
    print(f"has_image  : {bool(args.image)}")
    print(f"timeout    : {args.timeout}")
    print("=" * 80)

    start = time.time()
    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n请求失败，耗时 {elapsed:.3f}s")
        print(f"错误: {repr(e)}")
        raise

    elapsed = time.time() - start
    result = response_to_dict(response)

    print(f"\n请求成功，耗时 {elapsed:.3f}s")
    print("-" * 80)
    print("模型输出:\n")
    print(result["content"] or "")
    print("-" * 80)
    print("Usage:")
    print(json.dumps(result["usage"], ensure_ascii=False, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request": {
                "base_url": args.base_url,
                "model": args.model,
                "prompt": args.prompt,
                "image": args.image,
                "timeout": args.timeout,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "disable_thinking_extra_body": args.disable_thinking_extra_body,
            },
            "elapsed_seconds": elapsed,
            "response": result,
        }
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
