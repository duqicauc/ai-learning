import ssl
import httpx
from openai import OpenAI

def create_ssl_context():
    """Create SSL context that bypasses certificate verification"""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

def test_stream_chat():
    """Test streaming chat with proper error handling"""
    try:
        # Create HTTP client with SSL bypass
        http_client = httpx.Client(
            verify=False,  # Bypass SSL verification
            timeout=30.0
        )
        
        # Create OpenAI client with custom HTTP client
        client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-lohnuvviyzcltomzafjlnbghqzpjhlifyleenzrkfwxnlprd",
            http_client=http_client
        )
        
        print("🚀 Starting streaming chat demo...")
        print("🔗 Connecting to SiliconFlow API...")
        
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # Updated to a more stable model
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。"},
                {"role": "user", "content": "你好，请简单介绍一下你自己。"},
            ],
            stream=True,
            temperature=0.7,
            max_tokens=500
        )
        
        print("💬 AI回复：")
        full_response = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n\n✅ 流式对话演示完成！")
        print(f"📝 完整回复长度: {len(full_response)} 字符")
        
    except Exception as e:
        print(f"❌ 连接错误: {str(e)}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 验证API密钥是否有效")
        print("   3. 尝试使用VPN")
        
    finally:
        # Clean up HTTP client
        if 'http_client' in locals():
            http_client.close()

if __name__ == "__main__":
    test_stream_chat()
