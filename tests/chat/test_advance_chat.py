"""
高级多轮对话自动化测试脚本
演示token管理和对话压缩功能
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 添加src目录到Python路径
src_path = os.path.join(project_root, 'src', '07_chat')
sys.path.insert(0, src_path)

from chatMultiAdvanceDemo import AdvancedChatManager

def test_token_management_and_compression():
    """测试token管理和对话压缩功能"""
    print("🧪 高级多轮对话功能测试")
    print("="*60)
    
    # 创建聊天管理器，设置较低的阈值便于演示
    chat_manager = AdvancedChatManager(max_tokens=1500, summarize_interval=3)
    
    print(f"⚙️  配置参数:")
    print(f"   - 最大token数: {chat_manager.max_tokens}")
    print(f"   - 压缩间隔: 每{chat_manager.summarize_interval}轮")
    print(f"   - 压缩阈值: {chat_manager.max_tokens * 0.8} tokens (80%)")
    
    # 测试对话序列
    test_conversations = [
        "你好，我想学习人工智能和机器学习，请给我一个详细的学习路线图",
        "我应该先学习哪些数学基础知识？比如线性代数、微积分、概率论等",
        "Python在机器学习中有哪些重要的库？请详细介绍它们的用途和特点",
        "深度学习和传统机器学习有什么区别？我应该从哪个开始学习？",
        "能推荐一些实际的项目来练习机器学习技能吗？最好是从简单到复杂的",
        "在学习过程中，我应该如何跟上最新的AI技术发展和研究进展？",
        "如何评估一个机器学习模型的性能？有哪些常用的评估指标？"
    ]
    
    try:
        for i, user_message in enumerate(test_conversations, 1):
            print(f"\n{'='*20} 第 {i} 轮对话 {'='*20}")
            print(f"👤 用户: {user_message}")
            
            # 显示对话前的状态
            stats_before = chat_manager.get_conversation_stats()
            print(f"📊 对话前状态: {stats_before['current_tokens']} tokens ({stats_before['token_usage_percent']:.1f}%)")
            
            try:
                # 发送消息并获取回复
                ai_response = chat_manager.chat(user_message)
                print(f"🤖 AI回复: {ai_response[:150]}...")
                
                # 显示对话后的状态
                stats_after = chat_manager.get_conversation_stats()
                print(f"📊 对话后状态: {stats_after['current_tokens']} tokens ({stats_after['token_usage_percent']:.1f}%)")
                
                # 显示消息数量变化
                print(f"📝 消息数量: {stats_after['total_messages']} 条")
                
            except Exception as e:
                print(f"❌ 请求失败: {e}")
                print("⏭️  跳过此轮对话，继续测试...")
                continue
        
        print(f"\n{'='*60}")
        print("📊 最终测试结果")
        print("="*60)
        
        # 显示最终统计
        final_stats = chat_manager.get_conversation_stats()
        print(f"✅ 测试完成统计:")
        print(f"   - 成功对话轮数: {final_stats['total_rounds']}")
        print(f"   - 最终消息数: {final_stats['total_messages']}")
        print(f"   - 最终token数: {final_stats['current_tokens']}")
        print(f"   - Token使用率: {final_stats['token_usage_percent']:.1f}%")
        
        # 显示对话历史结构
        print(f"\n📚 最终对话历史结构:")
        chat_manager.show_history()
        
        # 验证压缩效果
        if final_stats['total_rounds'] >= chat_manager.summarize_interval:
            print(f"\n✅ 压缩功能验证:")
            print(f"   - 预期触发压缩: 是 (≥{chat_manager.summarize_interval}轮)")
            print(f"   - 实际消息数: {final_stats['total_messages']}")
            print(f"   - 压缩效果: {'有效' if final_stats['total_messages'] < final_stats['total_rounds'] * 2 else '待优化'}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        
    finally:
        chat_manager.close()
        print(f"\n🔚 测试结束，连接已关闭")

def test_token_counting():
    """测试token计数功能"""
    print("\n🧮 Token计数功能测试")
    print("-" * 40)
    
    chat_manager = AdvancedChatManager()
    
    # 测试不同长度的消息
    test_messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！我是AI助手，很高兴为您服务。"},
        {"role": "user", "content": "请详细解释一下什么是机器学习，包括它的定义、主要类型、应用领域以及与人工智能的关系。"},
    ]
    
    for i, msg in enumerate(test_messages, 1):
        tokens = chat_manager.count_tokens([msg])
        print(f"消息 {i}: {tokens} tokens")
        print(f"   内容: {msg['content'][:50]}...")
    
    # 测试整个对话的token数
    total_tokens = chat_manager.count_tokens(test_messages)
    print(f"\n总计: {total_tokens} tokens")
    
    chat_manager.close()

if __name__ == "__main__":
    print("🚀 启动高级多轮对话测试套件")
    print("="*60)
    
    # 测试1: Token计数功能
    test_token_counting()
    
    # 测试2: Token管理和压缩功能
    test_token_management_and_compression()
    
    print("\n🎉 所有测试完成！")