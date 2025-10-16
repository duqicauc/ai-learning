# AI Learning 项目结构规划

## 🎯 推荐的项目目录结构

```
ai-learning/
├── README.md                    # 项目主要说明文档
├── requirements.txt             # Python依赖包列表
├── .gitignore                  # Git忽略文件配置
├── LICENSE                     # 开源许可证
├── CONTRIBUTING.md             # 贡献指南
├── 
├── 📁 src/                     # 源代码目录
│   ├── 01_basics/              # 基础概念和入门代码
│   │   ├── tensor_operations.py
│   │   ├── linear_regression.py
│   │   └── iris_classification.py
│   │
│   ├── 02_neural_networks/     # 神经网络相关
│   │   ├── feedforward_nn.py
│   │   ├── activation_functions.py
│   │   └── optimization_demo.py
│   │
│   ├── 03_cnn/                 # 卷积神经网络
│   │   ├── basic_cnn.py
│   │   ├── cats_dogs_classifier.py
│   │   ├── improved_cnn.py
│   │   └── transfer_learning.py
│   │
│   ├── 04_rnn/                 # 循环神经网络
│   │   ├── basic_rnn.py
│   │   ├── lstm_demo.py
│   │   └── text_classification.py
│   │
│   ├── 05_transformers/        # Transformer架构
│   │   ├── attention_mechanism.py
│   │   ├── transformer_demo.py
│   │   └── bert_example.py
│   │
│   ├── 06_advanced/            # 高级主题
│   │   ├── gan_demo.py
│   │   ├── reinforcement_learning.py
│   │   └── model_optimization.py
│   │
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── data_loader.py
│       ├── visualization.py
│       ├── model_utils.py
│       └── training_utils.py
│
├── 📁 data/                    # 数据集目录
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理后的数据
│   ├── external/               # 外部数据集
│   └── README.md               # 数据说明文档
│
├── 📁 notebooks/               # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   ├── 03_visualization_analysis.ipynb
│   └── tutorials/              # 教程笔记本
│       ├── cnn_tutorial.ipynb
│       └── rnn_tutorial.ipynb
│
├── 📁 models/                  # 训练好的模型
│   ├── saved_models/           # 保存的模型文件
│   ├── checkpoints/            # 训练检查点
│   └── configs/                # 模型配置文件
│
├── 📁 experiments/             # 实验记录
│   ├── experiment_logs/        # 实验日志
│   ├── results/                # 实验结果
│   └── configs/                # 实验配置
│
├── 📁 docs/                    # 文档目录
│   ├── notes/                  # 学习笔记
│   │   ├── 张量学习笔记.md
│   │   ├── CNN原理笔记.md
│   │   └── 深度学习数学基础.md
│   ├── tutorials/              # 教程文档
│   ├── api/                    # API文档
│   └── images/                 # 文档图片
│
├── 📁 tests/                   # 测试代码
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_data_processing.py
│
├── 📁 scripts/                 # 脚本文件
│   ├── train_model.py          # 训练脚本
│   ├── evaluate_model.py       # 评估脚本
│   ├── data_preprocessing.py   # 数据预处理脚本
│   └── setup_environment.py    # 环境设置脚本
│
└── 📁 assets/                  # 静态资源
    ├── images/                 # 图片资源
    ├── videos/                 # 视频资源
    └── presentations/          # 演示文稿
```

## 📋 目录说明

### 核心目录

- **`src/`**: 按学习进度和主题组织的源代码
- **`data/`**: 数据集管理，区分原始数据和处理后数据
- **`notebooks/`**: 交互式学习和实验笔记本
- **`docs/`**: 完整的文档体系，包括笔记和教程

### 支持目录

- **`models/`**: 模型文件管理，便于版本控制和复用
- **`experiments/`**: 实验追踪，记录不同配置的结果
- **`tests/`**: 代码质量保证
- **`scripts/`**: 自动化脚本，提高工作效率

## 🔄 文件迁移计划

### 当前文件 → 新位置

```
src/cnncatsanddogs.py → src/03_cnn/cats_dogs_classifier.py
src/improved_cnn_v1.py → src/03_cnn/improved_cnn.py
src/transfer_learning_cnn.py → src/03_cnn/transfer_learning.py
src/cnnminsetdemo.py → src/03_cnn/basic_cnn.py
src/RNNDemo.py → src/04_rnn/basic_rnn.py
src/transfomerdemo.py → src/05_transformers/transformer_demo.py
src/irisdemo.py → src/01_basics/iris_classification.py
src/lineregdemo.py → src/01_basics/linear_regression.py
src/tensorregistdemo.py → src/01_basics/tensor_operations.py
src/tensorcomputedemo-teacher.py → src/01_basics/tensor_operations.py
src/GPUtraning.py → src/utils/training_utils.py
笔记/张量学习笔记.md → docs/notes/张量学习笔记.md
```

## 🎯 命名规范

### 文件命名
- 使用小写字母和下划线：`cats_dogs_classifier.py`
- 避免中文文件名，使用英文描述功能
- 类文件使用驼峰命名：`CatDogClassifier`

### 目录命名
- 使用数字前缀表示学习顺序：`01_basics/`, `02_neural_networks/`
- 使用描述性名称：`cnn/`, `rnn/`, `transformers/`
- 避免空格，使用下划线连接：`neural_networks/`

## 📚 版本控制建议

### Git 分支策略
- `main`: 稳定的主分支
- `develop`: 开发分支
- `feature/topic-name`: 特性分支
- `experiment/exp-name`: 实验分支

### 提交信息规范
```
feat: 添加CNN猫狗分类器
fix: 修复数据加载器内存泄漏
docs: 更新README文档
refactor: 重构模型训练代码
test: 添加模型测试用例
```