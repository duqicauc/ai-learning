"""
卷积神经网络(CNN)实现和应用

本模块包含各种CNN架构和图像处理应用：
- 基础CNN实现
- 图像分类任务
- 迁移学习
- 模型改进和优化
"""

__version__ = "1.0.0"
__author__ = "AI Learning Project"

# 导入主要模块
from . import basic_cnn
from . import cats_dogs_classifier
from . import improved_cnn
from . import transfer_learning

__all__ = [
    "basic_cnn",
    "cats_dogs_classifier",
    "improved_cnn", 
    "transfer_learning"
]