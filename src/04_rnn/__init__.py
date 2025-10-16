"""
循环神经网络(RNN)实现和应用

本模块包含各种RNN架构和序列处理应用：
- 基础RNN实现
- LSTM和GRU
- 序列预测
- 文本处理
"""

__version__ = "1.0.0"
__author__ = "AI Learning Project"

# 导入主要模块
from . import basic_rnn

__all__ = [
    "basic_rnn"
]