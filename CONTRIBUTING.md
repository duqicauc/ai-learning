# 🤝 贡献指南

感谢您对 AI Learning 项目的关注！我们欢迎所有形式的贡献，无论是代码、文档、bug报告还是功能建议。

## 📋 目录

- [贡献方式](#-贡献方式)
- [开发环境设置](#-开发环境设置)
- [代码规范](#-代码规范)
- [提交规范](#-提交规范)
- [Pull Request 流程](#-pull-request-流程)
- [问题反馈](#-问题反馈)

## 🎯 贡献方式

### 🐛 Bug 修复
- 修复现有代码中的错误
- 改进模型性能
- 优化内存使用

### ✨ 新功能
- 添加新的模型实现
- 增加数据处理工具
- 创建可视化功能

### 📚 文档改进
- 完善代码注释
- 编写教程文档
- 翻译现有文档

### 🧪 测试
- 编写单元测试
- 添加集成测试
- 性能基准测试

## 🛠️ 开发环境设置

### 1. Fork 和克隆项目

```bash
# Fork 项目到你的GitHub账户
# 然后克隆你的fork
git clone https://github.com/yourusername/ai-learning.git
cd ai-learning

# 添加上游仓库
git remote add upstream https://github.com/originalowner/ai-learning.git
```

### 2. 创建开发环境

```bash
# 创建虚拟环境
python -m venv ai-learning-dev
source ai-learning-dev/bin/activate  # Linux/macOS
# 或
ai-learning-dev\\Scripts\\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发工具
```

### 3. 安装开发工具

```bash
# 安装pre-commit钩子
pre-commit install

# 验证环境
python scripts/setup_environment.py
```

## 📝 代码规范

### Python 代码风格

我们使用以下工具确保代码质量：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

```bash
# 格式化代码
black src/ tests/
isort src/ tests/

# 检查代码
flake8 src/ tests/
mypy src/
```

### 命名规范

```python
# 文件名：小写+下划线
# 例如：cats_dogs_classifier.py

# 类名：驼峰命名
class CatDogClassifier:
    pass

# 函数名：小写+下划线
def train_model():
    pass

# 常量：大写+下划线
MAX_EPOCHS = 100

# 私有变量：下划线开头
_private_var = "internal use"
```

### 文档字符串

```python
def train_model(model, data_loader, epochs=10):
    """
    训练深度学习模型
    
    Args:
        model (nn.Module): 要训练的模型
        data_loader (DataLoader): 训练数据加载器
        epochs (int): 训练轮数，默认10
        
    Returns:
        dict: 包含训练历史的字典
        
    Raises:
        ValueError: 当epochs小于1时抛出
        
    Example:
        >>> model = CatDogCNN()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> history = train_model(model, loader, epochs=20)
    """
    pass
```

### 类型注解

```python
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

def process_batch(
    images: torch.Tensor, 
    labels: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """处理一个批次的数据"""
    pass

class ModelConfig:
    """模型配置类"""
    def __init__(
        self, 
        num_classes: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.5
    ) -> None:
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
```

## 📤 提交规范

### 提交信息格式

```
<类型>(<范围>): <描述>

<详细说明>

<相关Issue>
```

### 提交类型

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```bash
feat(cnn): 添加ResNet-50实现

- 实现ResNet-50架构
- 添加预训练权重加载
- 包含完整的训练和评估代码
- 添加性能基准测试

Closes #123
```

## 🔄 Pull Request 流程

### 1. 创建功能分支

```bash
# 确保主分支是最新的
git checkout main
git pull upstream main

# 创建新分支
git checkout -b feature/your-feature-name
```

### 2. 开发和测试

```bash
# 进行开发
# ...

# 运行测试
python -m pytest tests/
python -m pytest tests/test_your_feature.py -v

# 检查代码质量
black --check src/
flake8 src/
```

### 3. 提交更改

```bash
# 添加文件
git add .

# 提交
git commit -m "feat: 添加新功能描述"

# 推送到你的fork
git push origin feature/your-feature-name
```

### 4. 创建 Pull Request

1. 在GitHub上打开你的fork
2. 点击 "Compare & pull request"
3. 填写PR模板
4. 等待代码审查

### PR 检查清单

- [ ] 代码通过所有测试
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 遵循代码规范
- [ ] 提交信息清晰
- [ ] 没有合并冲突

## 🐛 问题反馈

### Bug 报告

使用以下模板报告Bug：

```markdown
**Bug描述**
简洁清晰地描述bug

**复现步骤**
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

**期望行为**
描述你期望发生什么

**实际行为**
描述实际发生了什么

**环境信息**
- OS: [例如 Windows 10]
- Python版本: [例如 3.8.5]
- PyTorch版本: [例如 1.9.0]
- CUDA版本: [例如 11.1]

**附加信息**
添加任何其他相关信息
```

### 功能请求

```markdown
**功能描述**
清晰简洁地描述你想要的功能

**动机**
解释为什么这个功能有用

**详细描述**
详细描述你希望如何实现这个功能

**替代方案**
描述你考虑过的其他解决方案

**附加信息**
添加任何其他相关信息或截图
```

## 🏷️ 标签系统

我们使用以下标签来分类Issues和PRs：

- `bug`: Bug报告
- `enhancement`: 功能增强
- `documentation`: 文档相关
- `good first issue`: 适合新手
- `help wanted`: 需要帮助
- `question`: 问题咨询
- `wontfix`: 不会修复

## 🎖️ 贡献者认可

我们会在以下地方认可贡献者：

- README.md 的贡献者列表
- 发布说明中的特别感谢
- 项目网站的贡献者页面

## 📞 联系方式

如果你有任何问题，可以通过以下方式联系我们：

- 创建Issue讨论
- 发送邮件到 maintainer@example.com
- 加入我们的Discord服务器

---

再次感谢您的贡献！🎉