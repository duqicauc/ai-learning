"""
基础AI/ML概念和实现

本模块包含深度学习的基础概念和入门级实现：
- 张量操作和计算
- 线性回归
- 分类算法
- 基础神经网络组件
"""

__version__ = "1.0.0"
__author__ = "AI Learning Project"

# 导入主要模块
from . import tensor_operations
from . import linear_regression
from . import iris_classification
from . import tensor_compute_tutorial

__all__ = [
    "tensor_operations",
    "linear_regression", 
    "iris_classification",
    "tensor_compute_tutorial"
]