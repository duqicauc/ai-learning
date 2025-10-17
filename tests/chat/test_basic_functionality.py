"""简单的功能验证测试"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 添加src目录到Python路径
src_path = os.path.join(project_root, 'src', '07_chat')
sys.path.insert(0, src_path)

from chatMultiAdvanceDemo import AdvancedChatManager

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 基本功能测试")
    print("="*40)
    
    # 创建聊天管理器
    chat_manager = AdvancedChatManager(max_tokens=1000, summarize_interval=2)
    print("✅ AdvancedChatManager 创建成功")
    
    # 测试token计数
    test_message = [{"role": "user", "content": "Hello, how are you?"}]
    token_count = chat_manager.count_tokens(test_message)
    print(f"✅ Token计数功能: {token_count} tokens")
    
    # 测试统计信息
    stats = chat_manager.get_conversation_stats()
    print(f"✅ 统计信息获取: {stats['current_tokens']} tokens, {stats['total_rounds']} 轮")
    
    # 测试添加消息
    chat_manager.add_user_message("测试消息")
    chat_manager.add_assistant_message("测试回复")
    
    updated_stats = chat_manager.get_conversation_stats()
    print(f"✅ 消息添加功能: {updated_stats['total_rounds']} 轮, {updated_stats['total_messages']} 条消息")
    
    # 关闭连接
    chat_manager.close()
    print("✅ 连接关闭成功")
    
    print("\n🎉 所有基本功能测试通过！")

if __name__ == "__main__":
    test_basic_functionality()