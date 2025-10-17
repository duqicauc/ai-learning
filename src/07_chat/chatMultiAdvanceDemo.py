import httpx
from openai import OpenAI
import tiktoken
from typing import List, Dict
import os

class AdvancedChatManager:
    """高级多轮对话管理器，支持token管理和对话精简"""
    
    def __init__(self, max_tokens=4000, summarize_interval=5):
        """
        初始化对话管理器
        
        Args:
            max_tokens: 最大token数限制
            summarize_interval: 每隔多少轮进行对话精简
        """
        self.max_tokens = max_tokens
        self.summarize_interval = summarize_interval
        self.conversation_history = []
        self.conversation_count = 0
        self.total_tokens_used = 0
        
        # 初始化token编码器
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            # 如果无法获取特定模型的编码器，使用通用编码器
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # 创建HTTP客户端
        self.http_client = httpx.Client(
            verify=False,
            timeout=30.0
        )
        
        # 配置API密钥
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            print("❌ 错误：请设置环境变量 SILICONFLOW_API_KEY")
            print("   可以通过以下方式设置：")
            print("   1. 创建 .env 文件并添加: SILICONFLOW_API_KEY=your-api-key")
            print("   2. 或在命令行中设置: set SILICONFLOW_API_KEY=your-api-key")
            return None
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_key,
            http_client=self.http_client
        )
        
        # 初始化系统消息
        self.system_message = {
            "role": "system", 
            "content": "你是一个专业的AI助手，请用中文回答问题。保持回答简洁明了。"
        }
        self.conversation_history.append(self.system_message)
    
    def count_tokens(self, messages: List[Dict]) -> int:
        """计算消息列表的token数量"""
        total_tokens = 0
        for message in messages:
            # 计算每条消息的token数
            message_tokens = len(self.encoding.encode(message["content"]))
            total_tokens += message_tokens
        return total_tokens
    
    def get_conversation_summary(self, messages: List[Dict]) -> str:
        """生成对话摘要"""
        try:
            # 构建摘要请求
            summary_messages = [
                {
                    "role": "system",
                    "content": "请将以下对话内容总结成简洁的摘要，保留关键信息和上下文。摘要应该在100字以内。"
                },
                {
                    "role": "user",
                    "content": f"请总结以下对话内容：\n{json.dumps(messages, ensure_ascii=False, indent=2)}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=summary_messages,
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️  摘要生成失败: {e}")
            # 如果摘要生成失败，返回简单的文本摘要
            return f"对话摘要：包含{len(messages)}条消息的对话内容"
    
    def compress_conversation_history(self):
        """压缩对话历史，保留系统消息和最近的对话"""
        if len(self.conversation_history) <= 3:  # 系统消息 + 至少一轮对话
            return
        
        print(f"\n🔄 正在压缩对话历史...")
        print(f"   压缩前：{len(self.conversation_history)} 条消息")
        
        # 保留系统消息
        system_msg = self.conversation_history[0]
        
        # 获取需要压缩的消息（除了系统消息和最近2轮对话）
        recent_messages = self.conversation_history[-4:]  # 保留最近2轮对话（4条消息）
        messages_to_compress = self.conversation_history[1:-4]  # 需要压缩的消息
        
        if messages_to_compress:
            # 生成摘要
            summary = self.get_conversation_summary(messages_to_compress)
            
            # 创建摘要消息
            summary_message = {
                "role": "system",
                "content": f"[对话历史摘要] {summary}"
            }
            
            # 重构对话历史：系统消息 + 摘要 + 最近的对话
            self.conversation_history = [system_msg, summary_message] + recent_messages
            
            print(f"   压缩后：{len(self.conversation_history)} 条消息")
            print(f"   📝 生成摘要：{summary[:50]}...")
        else:
            print("   无需压缩")
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.conversation_history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.conversation_history.append({"role": "assistant", "content": content})
        self.conversation_count += 1
        
        # 检查token使用情况
        current_tokens = self.count_tokens(self.conversation_history)
        usage_percent = (current_tokens / self.max_tokens) * 100
        
        print(f"\n📊 第{self.conversation_count}轮对话完成")
        print(f"   当前token数：{current_tokens}/{self.max_tokens} ({usage_percent:.1f}%)")
        
        # 压缩逻辑：优先级检查
        should_compress = False
        compress_reason = ""
        
        # 1. 紧急压缩：超过100%立即压缩
        if current_tokens > self.max_tokens:
            should_compress = True
            compress_reason = f"🚨 Token使用率超过100% ({usage_percent:.1f}%)"
        
        # 2. 预防性压缩：超过90%且达到压缩间隔
        elif (current_tokens > self.max_tokens * 0.9 and 
              self.conversation_count % self.summarize_interval == 0):
            should_compress = True
            compress_reason = f"⚠️  Token使用率过高 ({usage_percent:.1f}%) 且达到压缩间隔"
        
        # 3. 定期压缩：达到压缩间隔且超过80%
        elif (self.conversation_count % self.summarize_interval == 0 and 
              current_tokens > self.max_tokens * 0.8):
            should_compress = True
            compress_reason = f"🔄 定期压缩 ({usage_percent:.1f}%)"
        
        if should_compress:
            print(f"   {compress_reason}")
            self.compress_conversation_history()
    
    def get_conversation_stats(self) -> Dict:
        """获取对话统计信息"""
        current_tokens = self.count_tokens(self.conversation_history)
        return {
            "total_rounds": self.conversation_count,
            "total_messages": len(self.conversation_history) - 1,  # 减去系统消息
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "token_usage_percent": (current_tokens / self.max_tokens) * 100
        }
    
    def chat(self, user_input: str) -> str:
        """发送聊天请求并返回回复"""
        self.add_user_message(user_input)
        
        # 检查添加用户消息后是否需要紧急压缩
        current_tokens = self.count_tokens(self.conversation_history)
        if current_tokens > self.max_tokens:
            usage_percent = (current_tokens / self.max_tokens) * 100
            print(f"\n🚨 紧急压缩：添加用户消息后token使用率 {usage_percent:.1f}%")
            self.compress_conversation_history()
        
        try:
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            self.add_assistant_message(ai_response)
            
            return ai_response
            
        except Exception as e:
            # 如果请求失败，移除刚添加的用户消息
            self.conversation_history.pop()
            raise e
    
    def show_history(self):
        """显示对话历史"""
        stats = self.get_conversation_stats()
        print(f"\n📚 对话历史 (共{stats['total_messages']}条消息，{stats['current_tokens']} tokens)")
        print(f"   Token使用率: {stats['token_usage_percent']:.1f}%")
        print("-" * 50)
        
        for i, msg in enumerate(self.conversation_history[1:], 1):  # 跳过系统消息
            role = "您" if msg["role"] == "user" else "AI" if msg["role"] == "assistant" else "摘要"
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"   {i}. {role}: {content}")
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = [self.system_message]
        self.conversation_count = 0
        print("🗑️  对话历史已清空")
    
    def close(self):
        """关闭客户端连接"""
        if self.http_client:
            self.http_client.close()

def test_advanced_multi_turn_chat():
    """高级多轮对话演示"""
    chat_manager = AdvancedChatManager(max_tokens=2000, summarize_interval=3)  # 降低阈值便于演示
    
    try:
        print("🚀 高级多轮对话演示启动...")
        print("🔗 连接到 SiliconFlow API...")
        print("\n" + "="*60)
        print("💬 高级多轮对话模式 (智能Token管理)")
        print("="*60)
        print("📝 可用命令:")
        print("   - 直接输入文字进行对话")
        print("   - '/history' - 查看对话历史和token统计")
        print("   - '/stats' - 查看详细统计信息")
        print("   - '/clear' - 清空对话历史")
        print("   - '/help' - 显示帮助信息")
        print("   - '/quit' 或 '/exit' - 退出程序")
        print("="*60)
        print(f"⚙️  配置: 最大{chat_manager.max_tokens} tokens，每{chat_manager.summarize_interval}轮自动精简")
        
        while True:
            # 获取用户输入
            try:
                stats = chat_manager.get_conversation_stats()
                prompt = f"\n[第{stats['total_rounds'] + 1}轮] 您 ({stats['current_tokens']} tokens): "
                user_input = input(prompt).strip()
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
                    print("👋 再见！感谢使用高级多轮对话演示")
                    break
                elif command == '/help':
                    print("\n📖 帮助信息:")
                    print("   - 直接输入文字进行对话")
                    print("   - '/history' - 查看对话历史和token统计")
                    print("   - '/stats' - 查看详细统计信息")
                    print("   - '/clear' - 清空对话历史")
                    print("   - '/help' - 显示帮助信息")
                    print("   - '/quit' 或 '/exit' - 退出程序")
                    continue
                elif command == '/history':
                    chat_manager.show_history()
                    continue
                elif command == '/stats':
                    stats = chat_manager.get_conversation_stats()
                    print(f"\n📊 详细统计信息:")
                    print(f"   - 对话轮数: {stats['total_rounds']}")
                    print(f"   - 消息总数: {stats['total_messages']}")
                    print(f"   - 当前tokens: {stats['current_tokens']}")
                    print(f"   - 最大tokens: {stats['max_tokens']}")
                    print(f"   - 使用率: {stats['token_usage_percent']:.1f}%")
                    continue
                elif command == '/clear':
                    chat_manager.clear_history()
                    continue
                else:
                    print(f"❌ 未知命令: {user_input}")
                    print("💡 输入 '/help' 查看可用命令")
                    continue
            
            try:
                print("🤖 AI正在思考...")
                ai_response = chat_manager.chat(user_input)
                print(f"🤖 AI: {ai_response}")
                
            except Exception as e:
                print(f"❌ 请求失败: {str(e)}")
                print("💡 请检查网络连接或稍后重试")
        
        # 显示最终统计
        final_stats = chat_manager.get_conversation_stats()
        print(f"\n📊 会话结束统计:")
        print(f"   - 总对话轮数: {final_stats['total_rounds']}")
        print(f"   - 总消息数: {final_stats['total_messages']}")
        print(f"   - 最终token数: {final_stats['current_tokens']}")
        
    except Exception as e:
        print(f"❌ 初始化错误: {str(e)}")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 验证API密钥是否有效")
        print("   3. 尝试使用VPN")
        
    finally:
        chat_manager.close()

def test_auto_compression():
    """自动压缩功能测试"""
    print("🧪 自动压缩功能测试...")
    chat_manager = AdvancedChatManager(max_tokens=1000, summarize_interval=2)  # 更低的阈值
    
    try:
        # 模拟多轮对话
        test_conversations = [
            "你好，我想学习Python编程",
            "Python有哪些主要的应用领域？",
            "我应该从哪些基础知识开始学习？",
            "能推荐一些好的Python学习资源吗？",
            "学习Python大概需要多长时间？",
            "我在学习过程中遇到困难该怎么办？"
        ]
        
        for i, user_message in enumerate(test_conversations, 1):
            print(f"\n[测试轮次 {i}]")
            print(f"👤 用户: {user_message}")
            
            try:
                ai_response = chat_manager.chat(user_message)
                print(f"🤖 AI: {ai_response[:100]}...")
                
                # 显示当前状态
                stats = chat_manager.get_conversation_stats()
                print(f"📊 当前状态: {stats['current_tokens']} tokens ({stats['token_usage_percent']:.1f}%)")
                
            except Exception as e:
                print(f"❌ 请求失败: {e}")
                break
        
        # 显示最终历史
        print(f"\n📚 最终对话历史:")
        chat_manager.show_history()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        chat_manager.close()

def main():
    """主函数，提供选择菜单"""
    print("🎯 高级聊天演示程序")
    print("="*40)
    print("请选择演示模式:")
    print("1. 交互式高级多轮对话")
    print("2. 自动压缩功能测试")
    print("3. 退出程序")
    print("="*40)
    
    while True:
        try:
            choice = input("请输入选择 (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 启动交互式高级多轮对话...")
                test_advanced_multi_turn_chat()
                break
            elif choice == "2":
                print("\n🚀 启动自动压缩功能测试...")
                test_auto_compression()
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