import httpx
from openai import OpenAI

def test_multi_turn_conversation():
    """自动化测试多轮对话功能"""
    try:
        # 创建HTTP客户端，绕过SSL验证
        http_client = httpx.Client(
            verify=False,  # 绕过SSL证书验证
            timeout=30.0
        )
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-lohnuvviyzcltomzafjlnbghqzpjhlifyleenzrkfwxnlprd",
            http_client=http_client
        )
        
        print("🚀 多轮对话自动化测试启动...")
        print("🔗 连接到 SiliconFlow API...")
        print("\n" + "="*50)
        print("💬 多轮对话演示")
        print("="*50)
        
        # 初始化对话历史
        conversation_history = [
            {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。保持回答简洁明了。"}
        ]
        
        # 模拟多轮对话
        test_conversations = [
            "你好，请简单介绍一下你自己",
            "你刚才提到你是AI助手，那你能帮我做什么？",
            "我想学习Python编程，你有什么建议吗？",
            "谢谢你的建议！那我应该从哪个项目开始练习呢？"
        ]
        
        for i, user_message in enumerate(test_conversations, 1):
            print(f"\n[第{i}轮对话]")
            print(f"👤 用户: {user_message}")
            
            # 添加用户消息到历史
            conversation_history.append({"role": "user", "content": user_message})
            
            try:
                print("🤖 AI正在思考...")
                
                # 发送聊天请求
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=conversation_history,
                    temperature=0.7,
                    max_tokens=300
                )
                
                ai_response = response.choices[0].message.content
                
                # 添加AI回复到历史
                conversation_history.append({"role": "assistant", "content": ai_response})
                
                # 显示AI回复
                print(f"🤖 AI: {ai_response}")
                
            except Exception as e:
                print(f"❌ 请求失败: {str(e)}")
                # 移除刚添加的用户消息，因为请求失败了
                conversation_history.pop()
                break
        
        # 显示对话历史统计
        print(f"\n" + "="*50)
        print(f"📊 对话统计:")
        print(f"   - 总轮数: {len([msg for msg in conversation_history if msg['role'] == 'user'])}")
        print(f"   - 历史消息数: {len(conversation_history) - 1}")  # 减去system消息
        print(f"   - 上下文长度: {sum(len(msg['content']) for msg in conversation_history)} 字符")
        print("="*50)
        
        print("\n✅ 多轮对话演示完成！")
        print("💡 特点:")
        print("   - 保持对话上下文")
        print("   - AI能记住之前的对话内容")
        print("   - 支持连续的话题讨论")
        
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

if __name__ == "__main__":
    test_multi_turn_conversation()