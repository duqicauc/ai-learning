"""测试修复后的token压缩功能"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 添加src目录到Python路径
src_path = os.path.join(project_root, 'src', '07_chat')
sys.path.insert(0, src_path)

from chatMultiAdvanceDemo import AdvancedChatManager

def test_compression_triggers():
    """测试不同情况下的压缩触发机制"""
    print("🧪 测试token压缩触发机制")
    print("="*50)
    
    # 创建一个低阈值的聊天管理器便于测试
    chat_manager = AdvancedChatManager(max_tokens=500, summarize_interval=2)
    
    try:
        print("\n📝 测试场景1：正常对话，逐步增加token")
        
        # 模拟多轮对话，逐步增加token数量
        test_messages = [
            "你好，请介绍一下人工智能的发展历史",
            "请详细解释深度学习的工作原理和主要应用领域",
            "能否分析一下机器学习和深度学习的区别，以及它们各自的优缺点",
            "请描述一下自然语言处理技术的最新进展和未来发展趋势",
            "解释一下大语言模型的训练过程和技术挑战",
            "分析一下人工智能在医疗、金融、教育等领域的具体应用案例"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- 第{i}轮测试 ---")
            print(f"用户输入: {message[:30]}...")
            
            # 获取当前状态
            stats_before = chat_manager.get_conversation_stats()
            print(f"发送前: {stats_before['current_tokens']} tokens ({stats_before['token_usage_percent']:.1f}%)")
            
            # 模拟添加用户消息和AI回复
            chat_manager.add_user_message(message)
            
            # 模拟AI回复（较长的回复以增加token数）
            ai_response = f"这是第{i}轮的详细回复。" + "这是一个比较长的回复内容，用来增加token数量。" * 10
            chat_manager.add_assistant_message(ai_response)
            
            # 获取处理后状态
            stats_after = chat_manager.get_conversation_stats()
            print(f"处理后: {stats_after['current_tokens']} tokens ({stats_after['token_usage_percent']:.1f}%)")
            
            # 检查是否触发了压缩
            if stats_after['current_tokens'] < stats_before['current_tokens'] + 100:
                print("✅ 检测到压缩已触发")
            
            print(f"当前消息数: {len(chat_manager.conversation_history)}")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
    
    finally:
        chat_manager.close()

def test_emergency_compression():
    """测试紧急压缩功能（超过100%时立即压缩）"""
    print("\n\n🚨 测试紧急压缩功能")
    print("="*50)
    
    # 创建一个非常低阈值的管理器
    chat_manager = AdvancedChatManager(max_tokens=200, summarize_interval=5)
    
    try:
        # 添加一个很长的消息，强制超过100%
        long_message = "这是一个非常长的消息。" * 50  # 重复50次
        
        print(f"添加超长消息: {len(long_message)} 字符")
        
        stats_before = chat_manager.get_conversation_stats()
        print(f"添加前: {stats_before['current_tokens']} tokens")
        
        # 添加用户消息
        chat_manager.add_user_message(long_message)
        
        # 检查是否立即触发了压缩
        stats_after = chat_manager.get_conversation_stats()
        print(f"添加后: {stats_after['current_tokens']} tokens ({stats_after['token_usage_percent']:.1f}%)")
        
        if stats_after['token_usage_percent'] > 100:
            print("⚠️  仍然超过100%，可能需要进一步优化")
        else:
            print("✅ 紧急压缩成功触发")
            
    except Exception as e:
        print(f"❌ 紧急压缩测试失败: {str(e)}")
    
    finally:
        chat_manager.close()

def test_compression_intervals():
    """测试不同压缩间隔的效果"""
    print("\n\n🔄 测试压缩间隔设置")
    print("="*50)
    
    intervals = [2, 3, 5]
    
    for interval in intervals:
        print(f"\n--- 测试间隔: {interval}轮 ---")
        chat_manager = AdvancedChatManager(max_tokens=400, summarize_interval=interval)
        
        try:
            # 进行多轮对话
            for i in range(interval + 2):  # 超过间隔数
                message = f"第{i+1}轮测试消息，内容较长以增加token数量。" * 5
                chat_manager.add_user_message(message)
                
                ai_response = f"第{i+1}轮AI回复，同样较长以增加token数量。" * 8
                chat_manager.add_assistant_message(ai_response)
                
                stats = chat_manager.get_conversation_stats()
                print(f"  第{i+1}轮: {stats['current_tokens']} tokens ({stats['token_usage_percent']:.1f}%)")
                
        except Exception as e:
            print(f"❌ 间隔测试失败: {str(e)}")
        
        finally:
            chat_manager.close()

if __name__ == "__main__":
    print("🔧 Token压缩功能修复验证测试")
    print("="*60)
    
    test_compression_triggers()
    test_emergency_compression()
    test_compression_intervals()
    
    print("\n🎉 所有压缩测试完成！")