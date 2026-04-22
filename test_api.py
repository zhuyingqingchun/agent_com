#!/usr/bin/env python3
"""测试 API 是否正常调用"""

import os
import sys

# 检查环境变量
api_key = os.environ.get("VLM_API_KEY", "")
api_url = os.environ.get("DEBUG_API_URL", "https://ark.cn-beijing.volces.com/api/v3")
model_id = os.environ.get("DEBUG_MODEL_ID", "doubao-seed-1-6-vision-250815")

print("=" * 60)
print("API 配置检查")
print("=" * 60)
print(f"API_URL: {api_url}")
print(f"MODEL_ID: {model_id}")
print(f"VLM_API_KEY: {'已设置 (' + api_key[:10] + '...)' if api_key else '未设置'}")
print()

if not api_key:
    print("错误: VLM_API_KEY 未设置")
    sys.exit(1)

try:
    from openai import OpenAI
    
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )
    
    print("正在测试 API 调用...")
    
    # 简单的文本测试（不需要图片）
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "user", "content": "你好，这是一个测试消息，请回复'API测试成功'"}
        ],
        max_tokens=50
    )
    
    print()
    print("=" * 60)
    print("API 调用成功!")
    print("=" * 60)
    print(f"回复内容: {response.choices[0].message.content}")
    print(f"使用 tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    
except ImportError:
    print("错误: 未安装 openai 包，尝试安装...")
    os.system("pip install openai -q")
    print("请重新运行脚本")
    sys.exit(1)
    
except Exception as e:
    print()
    print("=" * 60)
    print("API 调用失败!")
    print("=" * 60)
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    sys.exit(1)
