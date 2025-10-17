#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉大模型完整演示程序
展示如何使用代码请求视觉大模型进行图像分析

功能特性:
1. 支持本地图像分析
2. 支持网络图像分析  
3. 多种问题类型演示
4. 完整的错误处理
5. 详细的代码示例
"""

import os
import base64
import requests
import urllib3
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import io

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

class VisionModelAPI:
    """视觉大模型API封装类"""
    
    def __init__(self):
        """初始化API客户端"""
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1"
        self.model = "Qwen/Qwen2-VL-72B-Instruct"  # 使用测试成功的模型
        
        if not self.api_key:
            raise ValueError("❌ 未找到API密钥，请检查.env文件")
        
        print(f"✅ 视觉大模型API初始化成功")
        print(f"🔑 API密钥: {self.api_key[:10]}...")
        print(f"🤖 使用模型: {self.model}")
    
    def encode_image_to_base64(self, image_path):
        """将本地图像编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ 图像编码失败: {e}")
            return None
    
    def download_and_encode_image(self, image_url):
        """下载网络图像并编码为base64"""
        try:
            print(f"📥 下载图像: {image_url}")
            response = requests.get(image_url, verify=False, timeout=10)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                print(f"❌ 下载失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 下载异常: {e}")
            return None
    
    def analyze_image(self, base64_image, question="请详细描述这张图片"):
        """分析图像"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            print(f"📡 发送API请求...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    print("❌ 响应格式异常")
                    return None
            else:
                print(f"❌ API请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return None

def create_demo_image():
    """创建演示图像"""
    # 创建一个更复杂的测试图像
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # 画太阳
    draw.ellipse([320, 30, 370, 80], fill='yellow', outline='orange', width=3)
    
    # 画云朵
    draw.ellipse([50, 40, 120, 80], fill='white', outline='lightgray')
    draw.ellipse([80, 30, 150, 70], fill='white', outline='lightgray')
    
    # 画房子
    draw.rectangle([150, 150, 250, 250], fill='brown', outline='black', width=2)
    draw.polygon([(140, 150), (200, 100), (260, 150)], fill='red', outline='darkred')
    
    # 画门
    draw.rectangle([180, 200, 220, 250], fill='black')
    
    # 画窗户
    draw.rectangle([160, 170, 190, 190], fill='lightblue', outline='blue')
    draw.rectangle([210, 170, 240, 190], fill='lightblue', outline='blue')
    
    # 画树
    draw.rectangle([280, 200, 300, 250], fill='brown')
    draw.ellipse([270, 150, 310, 210], fill='green', outline='darkgreen')
    
    # 保存图像
    image_path = "demo_scene.png"
    img.save(image_path)
    print(f"✅ 创建演示图像: {image_path}")
    return image_path

def demo_local_image_analysis():
    """演示本地图像分析"""
    print("\n🎯 本地图像分析演示")
    print("=" * 50)
    
    # 初始化API
    api = VisionModelAPI()
    
    # 创建演示图像
    image_path = create_demo_image()
    
    # 编码图像
    base64_image = api.encode_image_to_base64(image_path)
    if not base64_image:
        return
    
    # 测试不同类型的问题
    questions = [
        "请详细描述这张图片中的内容",
        "这张图片中有哪些颜色？",
        "图片中有几个建筑物？",
        "描述一下图片中的天气情况",
        "这张图片给你什么感觉？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        print("-" * 40)
        
        answer = api.analyze_image(base64_image, question)
        if answer:
            print(f"✅ 回答: {answer}")
        else:
            print("❌ 分析失败")
        print("-" * 40)

def demo_network_image_analysis():
    """演示网络图像分析"""
    print("\n🌐 网络图像分析演示")
    print("=" * 50)
    
    # 初始化API
    api = VisionModelAPI()
    
    # 测试网络图像
    test_urls = [
        "https://picsum.photos/400/300",  # 随机图像
        "https://httpbin.org/image/png"   # 测试PNG图像
    ]
    
    for url in test_urls:
        print(f"\n🔗 测试URL: {url}")
        
        # 下载并编码图像
        base64_image = api.download_and_encode_image(url)
        if not base64_image:
            print("❌ 图像下载失败，跳过")
            continue
        
        # 分析图像
        answer = api.analyze_image(base64_image, "请描述这张网络图片的内容")
        if answer:
            print(f"✅ 分析结果: {answer}")
            break  # 成功一个就够了
        else:
            print("❌ 分析失败")

def show_code_examples():
    """展示代码示例"""
    print("\n💻 完整代码示例")
    print("=" * 60)
    
    example_code = '''
# ===== 视觉大模型API调用完整示例 =====

import os
import base64
import requests
import urllib3
from dotenv import load_dotenv

# 1. 环境配置
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# 2. 获取API密钥
api_key = os.getenv("SILICONFLOW_API_KEY")

# 3. 图像编码函数
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 4. 视觉大模型API调用
def analyze_image_with_ai(image_path, question):
    # 编码图像
    base64_image = encode_image_to_base64(image_path)
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }}
            ]
        }],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    # 发送请求
    response = requests.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        headers=headers,
        json=data,
        verify=False,
        timeout=30
    )
    
    # 处理响应
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print(f"请求失败: {response.status_code}")
        return None

# 5. 使用示例
if __name__ == "__main__":
    # 分析本地图像
    result = analyze_image_with_ai("your_image.png", "请描述这张图片")
    print(f"AI分析结果: {result}")
    '''
    
    print(example_code)
    print("=" * 60)

def main():
    """主函数"""
    print("🎯 视觉大模型完整演示程序")
    print("=" * 60)
    print("📋 本程序将演示:")
    print("   1. 本地图像分析")
    print("   2. 网络图像分析")
    print("   3. 多种问题类型")
    print("   4. 完整代码示例")
    print("=" * 60)
    
    try:
        # 1. 本地图像分析演示
        demo_local_image_analysis()
        
        # 2. 网络图像分析演示
        demo_network_image_analysis()
        
        # 3. 展示代码示例
        show_code_examples()
        
        print("\n🎉 所有演示完成!")
        print("💡 您现在可以:")
        print("   - 使用提供的代码分析自己的图像")
        print("   - 修改问题来获得不同类型的分析")
        print("   - 集成到您的项目中")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())