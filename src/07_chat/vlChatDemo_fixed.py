"""
视觉大模型演示程序 - 修复版本
解决SSL证书验证问题，支持图像理解、图像描述、图像问答等功能
"""

import base64
import os
import requests
import ssl
import urllib3
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io
from typing import Optional, List, Dict

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 加载环境变量
load_dotenv()

class VisionChatDemo:
    """视觉大模型聊天演示类 - 修复版本"""
    
    def __init__(self, verify_ssl=False):
        """初始化视觉聊天演示"""
        # 配置API密钥
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("❌ 错误：请设置环境变量 SILICONFLOW_API_KEY")
            print("   可以通过以下方式设置：")
            print("   1. 创建 .env 文件并添加: SILICONFLOW_API_KEY=your-api-key")
            print("   2. 或在命令行中设置: set SILICONFLOW_API_KEY=your-api-key")
            raise ValueError("SILICONFLOW_API_KEY environment variable is required")
        
        # 创建OpenAI客户端，禁用SSL验证
        import httpx
        
        # 创建自定义的HTTP客户端，禁用SSL验证
        http_client = httpx.Client(verify=verify_ssl)
        
        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_key,
            http_client=http_client
        )
        
        # 支持的视觉模型
        self.vision_models = [
            "Qwen/Qwen2-VL-7B-Instruct",
            "OpenGVLab/InternVL2-26B",
            "meta-llama/Llama-3.2-11B-Vision-Instruct"
        ]
        
        self.current_model = self.vision_models[0]  # 默认使用Qwen2-VL
        self.verify_ssl = verify_ssl
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像文件编码为base64字符串"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"无法读取图像文件 {image_path}: {e}")
    
    def download_image_from_url(self, url: str, save_path: str = None) -> str:
        """从URL下载图像并保存到本地"""
        try:
            # 禁用SSL验证的请求
            response = requests.get(url, timeout=30, verify=self.verify_ssl)
            response.raise_for_status()
            
            if save_path is None:
                save_path = f"temp_image_{hash(url) % 10000}.jpg"
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ 图像已下载到: {save_path}")
            return save_path
        except Exception as e:
            raise Exception(f"下载图像失败: {e}")
    
    def analyze_image(self, image_path: str, question: str = "请描述这张图片") -> str:
        """分析图像并回答问题"""
        try:
            # 编码图像
            base64_image = self.encode_image_to_base64(image_path)
            
            # 构建消息
            messages = [
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
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ 图像分析失败: {e}"
    
    def analyze_image_from_url(self, image_url: str, question: str = "请描述这张图片") -> str:
        """从URL分析图像"""
        try:
            # 下载图像
            temp_path = self.download_image_from_url(image_url)
            
            # 分析图像
            result = self.analyze_image(temp_path, question)
            
            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            return f"❌ URL图像分析失败: {e}"
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            print("🔍 测试API连接...")
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10
            )
            print("✅ API连接测试成功！")
            return True
        except Exception as e:
            print(f"❌ API连接测试失败: {e}")
            return False
    
    def switch_model(self, model_index: int) -> bool:
        """切换视觉模型"""
        if 0 <= model_index < len(self.vision_models):
            self.current_model = self.vision_models[model_index]
            print(f"✅ 已切换到模型: {self.current_model}")
            return True
        else:
            print(f"❌ 无效的模型索引，请选择 0-{len(self.vision_models)-1}")
            return False
    
    def show_models(self):
        """显示可用的视觉模型"""
        print("\n📋 可用的视觉模型:")
        for i, model in enumerate(self.vision_models):
            current = " (当前)" if model == self.current_model else ""
            print(f"  {i}. {model}{current}")
    
    def interactive_demo(self):
        """交互式演示"""
        print("🎯 视觉大模型演示程序 - 修复版本")
        print("="*50)
        
        # 首先测试连接
        if not self.test_connection():
            print("❌ 无法连接到API，请检查网络设置")
            return
        
        print("📝 可用命令:")
        print("   - 'file <图片路径> [问题]' - 分析本地图片")
        print("   - 'url <图片URL> [问题]' - 分析网络图片")
        print("   - 'models' - 显示可用模型")
        print("   - 'switch <模型编号>' - 切换模型")
        print("   - 'test' - 测试连接")
        print("   - 'help' - 显示帮助")
        print("   - 'quit' - 退出程序")
        print("="*50)
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
                
                elif user_input.lower() == 'help':
                    print("\n📝 命令说明:")
                    print("   file <路径> [问题] - 分析本地图片，如: file image.jpg 这是什么动物？")
                    print("   url <URL> [问题] - 分析网络图片，如: url https://example.com/image.jpg")
                    print("   models - 显示所有可用的视觉模型")
                    print("   switch <编号> - 切换到指定编号的模型")
                    print("   test - 测试API连接")
                
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
                    
                    result = self.analyze_image(image_path, question)
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
                    
                    result = self.analyze_image_from_url(image_url, question)
                    print(f"\n💡 AI回复:\n{result}")
                
                else:
                    print("❌ 未知命令，输入 'help' 查看帮助")
            
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")

def main():
    """主函数"""
    print("🎯 视觉大模型演示程序 - 修复版本")
    print("="*50)
    print("⚠️  注意：此版本禁用了SSL验证以解决连接问题")
    print("请选择演示模式:")
    print("1. 交互式演示 (推荐)")
    print("2. 退出程序")
    print("="*50)
    
    while True:
        try:
            choice = input("请输入选择 (1-2): ").strip()
            
            if choice == "1":
                print("\n🚀 启动交互式演示...")
                vision_chat = VisionChatDemo(verify_ssl=False)
                vision_chat.interactive_demo()
                break
            elif choice == "2":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1 或 2")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()