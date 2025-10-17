import httpx
from openai import OpenAI

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
            api_key="sk-lohnuvviyzcltomzafjlnbghqzpjhlifyleenzrkfwxnlprd",
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

if __name__ == "__main__":
    test_simple_chat()
