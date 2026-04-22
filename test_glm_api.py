#!/usr/bin/env python3
"""测试 GLM-4V API 是否正常调用"""

import os
import sys
import base64
from pathlib import Path

# 使用用户提供的配置
api_key = "60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi"
api_url = "https://open.bigmodel.cn/api/paas/v4"
model_id = "glm-4v-flash"

print("=" * 60)
print("GLM-4V API 配置检查")
print("=" * 60)
print(f"API_URL: {api_url}")
print(f"MODEL_ID: {model_id}")
print(f"API_KEY: {'已设置 (' + api_key[:20] + '...)'}" if api_key else "未设置")
print()

if not api_key:
    print("错误: API_KEY 未设置")
    sys.exit(1)

try:
    from openai import OpenAI
    from PIL import Image
    
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_url
    )
    
    # 查找测试图片
    test_data_dir = Path("./test_data/offline")
    image_path = None
    
    if test_data_dir.exists():
        for subdir in test_data_dir.iterdir():
            if subdir.is_dir():
                for img_file in subdir.glob("*.png"):
                    image_path = img_file
                    break
            if image_path:
                break
    
    if not image_path:
        print("未找到测试图片，使用纯文本测试...")
        messages = [
            {"role": "user", "content": "你好，这是一个测试消息，请回复'GLM API测试成功'"}
        ]
    else:
        print(f"使用测试图片: {image_path}")
        
        # 读取并编码图片
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是什么应用界面？请简要描述。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
    
    print()
    print("正在测试 GLM-4V API 调用...")
    
    # 调用 API
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=200
    )
    
    print()
    print("=" * 60)
    print("GLM-4V API 调用成功!")
    print("=" * 60)
    print(f"回复内容: {response.choices[0].message.content}")
    if response.usage:
        print(f"输入 tokens: {response.usage.prompt_tokens}")
        print(f"输出 tokens: {response.usage.completion_tokens}")
        print(f"总 tokens: {response.usage.total_tokens}")
    
except ImportError as e:
    print(f"错误: 缺少依赖包 - {e}")
    print("尝试安装: pip install openai pillow")
    sys.exit(1)
    
except Exception as e:
    print()
    print("=" * 60)
    print("GLM-4V API 调用失败!")
    print("=" * 60)
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    sys.exit(1)
