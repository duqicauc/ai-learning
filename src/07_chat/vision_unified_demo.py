#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉大模型融合演示程序
结合自动演示和交互体验功能

功能特性:
1. 🎯 自动演示模式 - 展示完整的功能演示
2. 💬 交互体验模式 - 支持用户自定义图像分析
3. 🖼️ 支持本地和网络图像
4. 🤖 多模型支持和切换
5. 📝 详细的代码示例
6. 🔧 完整的错误处理
"""

import os
import base64
import requests
import urllib3
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
import io
from typing import Optional, List, Dict

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

class VisionUnifiedDemo:
    """视觉大模型融合演示类"""
    
    def __init__(self, verify_ssl=False):
        """初始化视觉演示程序"""
        # 配置API密钥
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            print("❌ 错误：请设置环境变量 SILICONFLOW_API_KEY")
            print("   可以通过以下方式设置：")
            print("   1. 创建 .env 文件并添加: SILICONFLOW_API_KEY=your-api-key")
            print("   2. 或在命令行中设置: set SILICONFLOW_API_KEY=your-api-key")
            raise ValueError("SILICONFLOW_API_KEY environment variable is required")
        
        # 创建OpenAI客户端，禁用SSL验证
        http_client = httpx.Client(verify=verify_ssl)
        
        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=self.api_key,
            http_client=http_client
        )
        
        # 基础配置
        self.base_url = "https://api.siliconflow.cn/v1"
        self.verify_ssl = verify_ssl
        
        # 支持的视觉模型
        self.vision_models = [
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct", 
            "deepseek-ai/deepseek-vl2",
            "OpenGVLab/InternVL2-26B",
            "OpenGVLab/InternVL2-8B"
        ]
        
        # 默认使用测试成功的模型
        self.current_model = self.vision_models[0]
        
        print(f"✅ 视觉大模型融合演示程序初始化成功")
        print(f"🔑 API密钥: {self.api_key[:10]}...")
        print(f"🤖 当前模型: {self.current_model}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将本地图像编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"图像编码失败: {e}")
    
    def download_and_encode_image(self, image_url: str) -> str:
        """下载网络图像并编码为base64"""
        try:
            response = requests.get(image_url, verify=self.verify_ssl, timeout=30)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise Exception(f"网络图像下载失败: {e}")
    
    def analyze_image_with_requests(self, base64_image: str, question: str = "请详细描述这张图片") -> str:
        """使用requests库分析图像（用于演示模式）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.current_model,
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                verify=self.verify_ssl,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ 分析失败: {e}"
    
    def analyze_image_with_openai(self, image_path: str, question: str = "请描述这张图片") -> str:
        """使用OpenAI客户端分析图像（用于交互模式）"""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 分析失败: {e}"
    
    def analyze_image_from_url_with_openai(self, image_url: str, question: str = "请描述这张图片") -> str:
        """使用OpenAI客户端分析网络图像（用于交互模式）"""
        try:
            base64_image = self.download_and_encode_image(image_url)
            
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 分析失败: {e}"
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            print("🔍 正在测试API连接...")
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                verify=self.verify_ssl,
                timeout=10
            )
            response.raise_for_status()
            print("✅ API连接测试成功")
            return True
        except Exception as e:
            print(f"❌ API连接测试失败: {e}")
            return False
    
    def switch_model(self, model_index: int) -> bool:
        """切换模型"""
        if 0 <= model_index < len(self.vision_models):
            self.current_model = self.vision_models[model_index]
            print(f"✅ 已切换到模型: {self.current_model}")
            return True
        else:
            print(f"❌ 无效的模型编号，请选择 0-{len(self.vision_models)-1}")
            return False
    
    def show_models(self):
        """显示可用模型"""
        print("\n🤖 可用的视觉模型:")
        for i, model in enumerate(self.vision_models):
            current = " (当前)" if model == self.current_model else ""
            print(f"   {i}. {model}{current}")
    
    def create_demo_image(self) -> str:
        """创建演示图像"""
        # 创建一个简单的演示图像
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # 绘制一个房子
        # 房子主体
        draw.rectangle([150, 180, 250, 250], fill='brown', outline='black', width=2)
        
        # 屋顶
        draw.polygon([(130, 180), (200, 120), (270, 180)], fill='red', outline='black')
        
        # 门
        draw.rectangle([180, 210, 220, 250], fill='black', outline='black')
        
        # 窗户
        draw.rectangle([160, 190, 190, 210], fill='yellow', outline='black', width=2)
        draw.rectangle([210, 190, 240, 210], fill='yellow', outline='black', width=2)
        
        # 太阳
        draw.ellipse([320, 50, 370, 100], fill='yellow', outline='orange', width=2)
        
        # 树
        draw.rectangle([80, 200, 100, 250], fill='brown')  # 树干
        draw.ellipse([60, 150, 120, 210], fill='green')    # 树冠
        
        # 保存图像
        image_path = "demo_unified_image.png"
        img.save(image_path)
        print(f"✅ 演示图像已创建: {image_path}")
        return image_path
    
    def demo_mode(self):
        """自动演示模式"""
        print("\n" + "="*60)
        print("🎯 自动演示模式")
        print("="*60)
        
        # 首先测试连接
        if not self.test_connection():
            print("❌ 无法连接到API，演示终止")
            return
        
        # 创建演示图像
        print("\n📸 步骤1: 创建演示图像")
        image_path = self.create_demo_image()
        
        # 编码图像
        print("\n🔄 步骤2: 编码图像为base64")
        try:
            base64_image = self.encode_image_to_base64(image_path)
            print("✅ 图像编码成功")
        except Exception as e:
            print(f"❌ 图像编码失败: {e}")
            return
        
        # 多种问题演示
        questions = [
            "请详细描述这张图片中的内容",
            "图片中有哪些颜色？",
            "图片中有几栋建筑物？",
            "这张图片适合用在什么场景？"
        ]
        
        print(f"\n🤖 步骤3: 使用模型 {self.current_model} 进行分析")
        print("-" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            print("🤖 AI正在分析...")
            
            result = self.analyze_image_with_requests(base64_image, question)
            print(f"💡 AI回复: {result}")
            print("-" * 30)
        
        # 显示核心代码示例
        self.show_code_examples()
        
        print("\n✅ 自动演示完成！")
    
    def interactive_mode(self):
        """交互体验模式"""
        print("\n" + "="*60)
        print("💬 交互体验模式")
        print("="*60)
        
        # 首先测试连接
        if not self.test_connection():
            print("❌ 无法连接到API，请检查网络设置")
            return
        
        print("\n📝 可用命令:")
        print("   - 'file <图片路径> [问题]' - 分析本地图片")
        print("   - 'url <图片URL> [问题]' - 分析网络图片")
        print("   - 'models' - 显示可用模型")
        print("   - 'switch <模型编号>' - 切换模型")
        print("   - 'test' - 测试连接")
        print("   - 'demo' - 切换到演示模式")
        print("   - 'help' - 显示帮助")
        print("   - 'quit' - 退出程序")
        print("="*60)
        print(f"🤖 当前模型: {self.current_model}")
        
        while True:
            try:
                user_input = input("\n💬 请输入命令: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                elif user_input.lower() == 'test':
                    self.test_connection()
                
                elif user_input.lower() == 'demo':
                    self.demo_mode()
                    print(f"\n💬 返回交互模式，当前模型: {self.current_model}")
                
                elif user_input.lower() == 'help':
                    print("\n📝 命令说明:")
                    print("   file <路径> [问题] - 分析本地图片，如: file image.jpg 这是什么动物？")
                    print("   url <URL> [问题] - 分析网络图片，如: url https://example.com/image.jpg")
                    print("   models - 显示所有可用的视觉模型")
                    print("   switch <编号> - 切换到指定编号的模型")
                    print("   test - 测试API连接")
                    print("   demo - 运行自动演示")
                
                elif user_input.lower() == 'models':
                    self.show_models()
                
                elif user_input.lower().startswith('switch '):
                    try:
                        model_index = int(user_input.split()[1])
                        self.switch_model(model_index)
                    except (IndexError, ValueError):
                        print("❌ 请提供有效的模型编号，如: switch 0")
                
                elif user_input.lower().startswith('file '):
                    parts = user_input.split(' ', 2)
                    if len(parts) < 2:
                        print("❌ 请提供图片路径，如: file image.jpg")
                        continue
                    
                    image_path = parts[1]
                    question = parts[2] if len(parts) > 2 else "请详细描述这张图片"
                    
                    if not os.path.exists(image_path):
                        print(f"❌ 文件不存在: {image_path}")
                        continue
                    
                    print(f"🔍 正在分析图片: {image_path}")
                    print(f"❓ 问题: {question}")
                    print("🤖 AI正在思考...")
                    
                    result = self.analyze_image_with_openai(image_path, question)
                    print(f"\n💡 AI回复:\n{result}")
                
                elif user_input.lower().startswith('url '):
                    parts = user_input.split(' ', 2)
                    if len(parts) < 2:
                        print("❌ 请提供图片URL，如: url https://example.com/image.jpg")
                        continue
                    
                    image_url = parts[1]
                    question = parts[2] if len(parts) > 2 else "请详细描述这张图片"
                    
                    print(f"🔍 正在分析网络图片: {image_url}")
                    print(f"❓ 问题: {question}")
                    print("🤖 AI正在思考...")
                    
                    result = self.analyze_image_from_url_with_openai(image_url, question)
                    print(f"\n💡 AI回复:\n{result}")
                
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")
            
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def show_code_examples(self):
        """显示核心代码示例"""
        print("\n" + "="*60)
        print("📝 核心代码示例")
        print("="*60)
        
        print("\n🔧 1. 基础API调用代码:")
        print("```python")
        print("import base64")
        print("import requests")
        print("")
        print("# 编码图像")
        print("def encode_image(image_path):")
        print("    with open(image_path, 'rb') as f:")
        print("        return base64.b64encode(f.read()).decode('utf-8')")
        print("")
        print("# API请求")
        print("def analyze_image(base64_image, question):")
        print("    headers = {")
        print("        'Authorization': f'Bearer {api_key}',")
        print("        'Content-Type': 'application/json'")
        print("    }")
        print("    ")
        print("    data = {")
        print("        'model': 'Qwen/Qwen2-VL-72B-Instruct',")
        print("        'messages': [{")
        print("            'role': 'user',")
        print("            'content': [")
        print("                {'type': 'text', 'text': question},")
        print("                {'type': 'image_url', 'image_url': {")
        print("                    'url': f'data:image/jpeg;base64,{base64_image}'")
        print("                }}")
        print("            ]")
        print("        }]")
        print("    }")
        print("    ")
        print("    response = requests.post(")
        print("        'https://api.siliconflow.cn/v1/chat/completions',")
        print("        headers=headers, json=data, verify=False")
        print("    )")
        print("    return response.json()['choices'][0]['message']['content']")
        print("```")
        
        print("\n🔧 2. OpenAI客户端代码:")
        print("```python")
        print("from openai import OpenAI")
        print("import httpx")
        print("")
        print("# 创建客户端（禁用SSL验证）")
        print("client = OpenAI(")
        print("    base_url='https://api.siliconflow.cn/v1',")
        print("    api_key=api_key,")
        print("    http_client=httpx.Client(verify=False)")
        print(")")
        print("")
        print("# 分析图像")
        print("response = client.chat.completions.create(")
        print("    model='Qwen/Qwen2-VL-72B-Instruct',")
        print("    messages=[{")
        print("        'role': 'user',")
        print("        'content': [")
        print("            {'type': 'text', 'text': question},")
        print("            {'type': 'image_url', 'image_url': {")
        print("                'url': f'data:image/jpeg;base64,{base64_image}'")
        print("            }}")
        print("        ]")
        print("    }]")
        print(")")
        print("```")
        
        print("\n💡 关键要点:")
        print("   • 使用 verify=False 解决SSL证书问题")
        print("   • 推荐使用 Qwen/Qwen2-VL-72B-Instruct 模型")
        print("   • 图像需要base64编码")
        print("   • 支持本地和网络图像")

def main():
    """主函数"""
    print("视觉大模型融合演示程序")
    print("="*60)
    print("本程序融合了自动演示和交互体验功能")
    print("支持多种视觉模型和图像分析任务")
    print("="*60)
    
    try:
        # 初始化演示程序
        demo = VisionUnifiedDemo()
        
        print("\n请选择运行模式:")
        print("   1. 自动演示模式 - 完整功能展示")
        print("   2. 交互体验模式 - 自定义图像分析")
        print("   3. 混合模式 - 先演示后交互")
        print("   4. 退出程序")
        
        while True:
            try:
                choice = input("\n请选择模式 (1-4): ").strip()
                
                if choice == '1':
                    demo.demo_mode()
                    break
                elif choice == '2':
                    demo.interactive_mode()
                    break
                elif choice == '3':
                    print("\n混合模式：先运行自动演示，然后进入交互模式")
                    demo.demo_mode()
                    print("\n" + "="*60)
                    print("自动演示完成，现在进入交互模式")
                    demo.interactive_mode()
                    break
                elif choice == '4':
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择，请输入 1-4")
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
    
    except Exception as e:
        print(f"❌ 程序初始化失败: {e}")
        print("请检查:")
        print("   1. 是否正确设置了 SILICONFLOW_API_KEY 环境变量")
        print("   2. 网络连接是否正常")
        print("   3. 是否安装了所需的依赖包")

if __name__ == "__main__":
    main()