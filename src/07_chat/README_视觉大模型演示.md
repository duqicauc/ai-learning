# 视觉大模型演示

本目录包含视觉大模型的演示程序，展示如何使用SiliconFlow API进行图像理解和分析。

## 文件说明

### 核心演示文件
- `vision_unified_simple.py` - **推荐使用** 融合版本的视觉大模型演示程序，支持自动演示和交互体验
- `vision_final_demo.py` - 完整的视觉大模型演示程序，包含多种功能展示
- `vlChatDemo_fixed.py` - 修复版本的视觉聊天演示，解决了SSL证书验证问题

### 其他聊天演示文件
- `chatDemo.py` - 基础聊天演示
- `chatDemo_fixed.py` - 修复版本的聊天演示
- `chatDemo_stream.py` - 流式聊天演示
- `chatDemo_stream_fixed.py` - 修复版本的流式聊天演示

## 功能特性

### vision_unified_simple.py（推荐）
- **融合设计**：结合自动演示和交互体验功能
- **三种运行模式**：
  - 自动演示模式：完整功能展示
  - 交互体验模式：自定义图像分析
  - 混合模式：先演示后交互
- **多模型支持**：支持5种视觉大模型切换
- **完整功能**：
  - 本地和网络图像分析
  - 多种问题类型演示
  - 详细的错误处理
  - 实时模型切换
  - 连接状态测试
- **用户友好**：简洁的命令行界面，避免编码问题

### vision_final_demo.py
- 支持本地和网络图像分析
- 多种问题类型演示
- 详细的错误处理
- 完整的代码示例展示
- 使用Qwen/Qwen2-VL-72B-Instruct模型

### vlChatDemo_fixed.py
- 交互式图像分析体验
- 支持图像理解、描述和问答
- 解决SSL证书验证问题
- 使用OpenAI客户端进行API调用

## 环境配置

1. 安装依赖包：
```bash
pip install openai requests pillow python-dotenv httpx urllib3
```

2. 设置API密钥：
创建 `.env` 文件并添加：
```
SILICONFLOW_API_KEY=your-api-key-here
```

## 运行演示

### 融合版本演示（推荐）
```bash
python vision_unified_simple.py
```
然后选择运行模式：
- 输入 `1` - 自动演示模式
- 输入 `2` - 交互体验模式  
- 输入 `3` - 混合模式
- 输入 `4` - 退出程序

### 完整功能演示
```bash
python vision_final_demo.py
```

### 交互式体验
```bash
python vlChatDemo_fixed.py
```

## 支持的视觉模型

程序支持以下视觉大模型：
1. Qwen/Qwen2-VL-72B-Instruct（默认）
2. Qwen/Qwen2-VL-7B-Instruct
3. deepseek-ai/deepseek-vl2
4. OpenGVLab/InternVL2-26B
5. OpenGVLab/InternVL2-8B

## 交互模式命令

在交互模式下，支持以下命令：
- `file <图片路径> [问题]` - 分析本地图片
- `models` - 显示可用模型
- `switch <模型编号>` - 切换模型
- `test` - 测试连接
- `demo` - 切换到演示模式
- `help` - 显示帮助
- `quit` - 退出程序

## 注意事项

1. 确保已正确设置SILICONFLOW_API_KEY环境变量
2. 程序会自动禁用SSL警告以避免证书验证问题
3. 支持多种图像格式（PNG、JPG、JPEG等）
4. 建议使用稳定的网络连接以确保API调用成功
5. 推荐使用 `vision_unified_simple.py` 获得最佳体验

## 项目结构

```
07_chat/
├── vision_unified_simple.py    # 融合版本演示程序（推荐）
├── vision_final_demo.py        # 完整功能演示
├── vlChatDemo_fixed.py         # 交互式聊天演示
├── chatDemo.py                 # 基础聊天演示
├── chatDemo_fixed.py           # 修复版聊天演示
├── chatDemo_stream.py          # 流式聊天演示
├── chatDemo_stream_fixed.py    # 修复版流式聊天演示
└── README_视觉大模型演示.md    # 本文档
```

## 更新日志

- **最新版本**：创建了 `vision_unified_simple.py` 融合版本
  - 结合了 `vision_final_demo.py` 和 `vlChatDemo_fixed.py` 的功能
  - 支持三种运行模式：自动演示、交互体验、混合模式
  - 解决了Unicode编码问题，提供更好的用户体验
  - 支持多模型切换和实时连接测试