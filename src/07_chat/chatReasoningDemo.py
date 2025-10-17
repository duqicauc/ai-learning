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
        
        print("开始思维链推理演示...")
        print("连接到 SiliconFlow API...")
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",  # Updated to a more stable model
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手，请用中文回答问题。"},
                {"role": "user", "content": "你好，请简单介绍一下你自己。"},
            ],
            stream=True,
            temperature=0.7,
            max_tokens=500
        )
        
        print("\n=== 思考过程 ===")
        reasoning_content = ""
        content = ""
        reasoning_started = False
        
        # 遍历流式响应的每个数据块
        for chunk in response:
            # 处理思考过程内容（reasoning_content）
            if chunk.choices[0].delta.reasoning_content:
                # 如果是第一次收到思考内容，标记思考阶段开始
                if not reasoning_started:
                    reasoning_started = True
                # 累积思考过程的完整内容
                reasoning_content += chunk.choices[0].delta.reasoning_content
                # 实时输出思考过程，不换行，立即刷新缓冲区
                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
            
            # 处理最终回答内容（content）
            elif chunk.choices[0].delta.content:
                # 如果之前有思考过程，现在开始输出最终回答，需要添加分隔标题
                if reasoning_started:
                    print("\n\n=== 最终回答 ===")
                    reasoning_started = False  # 标记思考阶段结束
                # 累积最终回答的完整内容
                content += chunk.choices[0].delta.content
                # 实时输出最终回答，不换行，立即刷新缓冲区
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print(f"\n\n演示完成。思考过程长度: {len(reasoning_content)} 字符，回答长度: {len(content)} 字符")
        
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
