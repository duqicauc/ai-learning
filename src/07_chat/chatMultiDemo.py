import httpx
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_multi_turn_chat():
    """多轮对话演示，支持连续对话和对话历史管理"""
    try:
        # 配置API密钥
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("❌ 错误：请设置环境变量 SILICONFLOW_API_KEY")
            print("   可以通过以下方式设置：")
            print("   1. 创建 .env 文件并添加: SILICONFLOW_API_KEY=your-api-key")
            print("   2. 或在命令行中设置: set SILICONFLOW_API_KEY=your-api-key")
            return
        
        # 创建HTTP客户端，绕过SSL验证
        http_client = httpx.Client(
            verify=False,  # 绕过SSL证书验证
            timeout=30.0
        )
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_key,
            http_client=http_client
        )
        
        print("🚀 多轮对话演示启动...")
        print("🔗 连接到 SiliconFlow API...")
        print("\n" + "="*50)
        print("💬 多轮对话模式")
        print("="*50)
        print("📝 可用命令:")
        print("   - 直接输入文字进行对话")
        print("   - '/history' - 查看对话历史")
        print("   - '/clear' - 清空对话历史")
        print("   - '/help' - 显示帮助信息")
        print("   - '/quit' 或 '/exit' - 退出程序")
        print("="*50)
        
        # 初始化对话历史
        conversation_history = [
            {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。保持回答简洁明了。"}
        ]
        
        conversation_count = 0
        
        while True:
            # 获取用户输入
            try:
                user_input = input(f"\n[第{conversation_count + 1}轮] 您: ").strip()
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except EOFError:
                print("\n\n👋 输入结束，正在退出...")
                break
            
            # 处理空输入
            if not user_input:
                print("⚠️  请输入有效内容")
                continue
            
            # 处理命令
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command in ['/quit', '/exit']:
                    print("👋 再见！感谢使用多轮对话演示")
                    break
                elif command == '/help':
                    print("\n📖 帮助信息:")
                    print("   - 直接输入文字进行对话")
                    print("   - '/history' - 查看对话历史")
                    print("   - '/clear' - 清空对话历史")
                    print("   - '/help' - 显示帮助信息")
                    print("   - '/quit' 或 '/exit' - 退出程序")
                    continue
                elif command == '/history':
                    print(f"\n📚 对话历史 (共{len(conversation_history)-1}条消息):")
                    for i, msg in enumerate(conversation_history[1:], 1):  # 跳过system消息
                        role = "您" if msg["role"] == "user" else "AI"
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"   {i}. {role}: {content}")
                    continue
                elif command == '/clear':
                    conversation_history = [conversation_history[0]]  # 保留system消息
                    conversation_count = 0
                    print("🗑️  对话历史已清空")
                    continue
                else:
                    print(f"❌ 未知命令: {user_input}")
                    print("💡 输入 '/help' 查看可用命令")
                    continue
            
            # 添加用户消息到历史
            conversation_history.append({"role": "user", "content": user_input})
            
            try:
                print("🤖 AI正在思考...")
                
                # 发送聊天请求
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=500
                )
                
                ai_response = response.choices[0].message.content
                
                # 添加AI回复到历史
                conversation_history.append({"role": "assistant", "content": ai_response})
                
                # 显示AI回复
                print(f"🤖 AI: {ai_response}")
                
                conversation_count += 1
                
            except Exception as e:
                print(f"❌ 请求失败: {str(e)}")
                print("💡 请检查网络连接或稍后重试")
                # 移除刚添加的用户消息，因为请求失败了
                conversation_history.pop()
        
    except Exception as e:
        print(f"❌ 初始化错误: {str(e)}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 验证API密钥是否有效")
        print("   3. 尝试使用VPN")
        
    finally:
        # 清理HTTP客户端
        if 'http_client' in locals():
            http_client.close()

def test_simple_chat():
    """测试简单的聊天功能，包含SSL修复和错误处理"""
    try:
        # 创建HTTP客户端，绕过SSL验证
        http_client = httpx.Client(
            verify=False,  # 绕过SSL证书验证
            timeout=30.0
        )
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_key,
            http_client=http_client
        )
        
        print("🚀 开始聊天演示...")
        print("🔗 连接到 SiliconFlow API...")
        
        # 发送聊天请求
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # 使用更稳定的模型
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。"},
                {"role": "user", "content": "你好，请简单介绍一下你自己。"},
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        print("💬 AI回复：")
        print(response.choices[0].message.content)
        print("\n✅ 聊天演示完成！")
        
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 验证API密钥是否有效")
        print("   3. 尝试使用VPN")
        
    finally:
        # 清理HTTP客户端
        if 'http_client' in locals():
            http_client.close()

def main():
    """主函数，提供选择菜单"""
    print("🎯 聊天演示程序")
    print("="*30)
    print("请选择演示模式:")
    print("1. 单轮对话演示")
    print("2. 多轮对话演示")
    print("3. 退出程序")
    print("="*30)
    
    while True:
        try:
            choice = input("请输入选择 (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 启动单轮对话演示...")
                test_simple_chat()
                break
            elif choice == "2":
                print("\n🚀 启动多轮对话演示...")
                test_multi_turn_chat()
                break
            elif choice == "3":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序已退出")
            break
        except EOFError:
            print("\n\n👋 程序已退出")
            break

if __name__ == "__main__":
    main()