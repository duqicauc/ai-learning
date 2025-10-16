# Trae + AutoDL 高效开发训练工作流指南

## 📋 目录
- [工作流概述](#工作流概述)
- [环境配置](#环境配置)
- [本地开发流程](#本地开发流程)
- [代码同步部署](#代码同步部署)
- [AutoDL训练管理](#autodl训练管理)
- [监控和调试](#监控和调试)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)

## 🎯 工作流概述

### 核心理念
- **本地开发**: 在Trae中进行代码编写、调试和快速验证
- **云端训练**: 在AutoDL上进行大规模模型训练
- **自动化同步**: 一键部署代码到训练环境
- **实时监控**: 远程监控训练进度和结果

### 工作流架构
```
Trae (本地开发)          AutoDL (云端训练)
├── 代码编写              ├── 模型训练
├── 快速测试              ├── 大数据处理
├── 配置管理              ├── GPU加速
└── 版本控制              └── 结果存储
        │                        ▲
        └── 自动同步 ──────────────┘
```

## ⚙️ 环境配置

### 1. 本地环境设置 (Trae)

#### 安装依赖
```bash
# 安装本地开发依赖
python scripts/local_dev.py --action setup
```

#### 配置文件
- `configs/local.yaml` - 本地开发配置
- `configs/sync_config.yaml` - 同步配置
- `requirements/local.txt` - 本地依赖

### 2. AutoDL环境设置

#### 初始化脚本
```bash
# 在AutoDL实例上运行
bash scripts/setup_autodl.sh
```

#### 配置SSH密钥
```bash
# 生成SSH密钥对
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 将公钥添加到AutoDL
cat ~/.ssh/id_rsa.pub
```

#### 更新同步配置
编辑 `configs/sync_config.yaml`:
```yaml
autodl:
  host: "your-autodl-instance.com"  # 替换为实际地址
  username: "root"
  key_file: "~/.ssh/id_rsa"
```

## 🔧 本地开发流程

### 1. 快速开始
```bash
# 设置开发环境
python scripts/local_dev.py --action setup

# 生成样本数据用于快速测试
python scripts/local_dev.py --action sample --sample-size 20

# 运行快速测试
python scripts/local_dev.py --action test --model fruits
```

### 2. 代码开发循环

#### 步骤1: 编写代码
在Trae中编辑模型代码:
- `src/03_cnn/fruits_classifier.py`
- `src/utils/data_preprocessing.py`
- `src/utils/model_utils.py`

#### 步骤2: 本地验证
```bash
# 代码质量检查
python scripts/local_dev.py --action lint

# 快速功能测试
python scripts/local_dev.py --action test --model fruits

# 数据验证
python scripts/local_dev.py --action validate
```

#### 步骤3: 配置调整
```bash
# 生成调试配置
python scripts/local_dev.py --action config --config-template debug

# 生成本地配置
python scripts/local_dev.py --action config --config-template local
```

### 3. 开发技巧

#### 使用样本数据
```bash
# 创建小规模样本用于快速迭代
python scripts/local_dev.py --action sample --sample-size 10
```

#### 配置模板
- `local.yaml` - 本地开发 (CPU, 小批次)
- `debug.yaml` - 调试模式 (最小配置)
- `autodl.yaml` - 云端训练 (GPU, 大批次)

## 🚀 代码同步部署

### 1. 一键同步和训练
```bash
# 完整流程: Git提交 → 代码同步 → 环境检查 → 启动训练
python scripts/sync_to_autodl.py --action full --training-config autodl
```

### 2. 分步操作

#### 仅同步代码
```bash
python scripts/sync_to_autodl.py --action sync
```

#### 仅启动训练
```bash
python scripts/sync_to_autodl.py --action train --training-config autodl
```

#### 监控训练
```bash
python scripts/sync_to_autodl.py --action monitor
```

### 3. 同步配置

#### 自动Git操作
```yaml
git:
  auto_commit: true
  commit_message_template: "sync: 同步代码到AutoDL {timestamp}"
  auto_push: true
```

#### 排除文件
```yaml
sync:
  exclude_patterns:
    - "*.pyc"
    - "__pycache__/"
    - "outputs/"
    - "data/"
    - "*.log"
```

## 🏋️ AutoDL训练管理

### 1. 训练配置

#### GPU优化配置 (`configs/autodl.yaml`)
```yaml
training:
  epochs: 100
  batch_size: 64
  device: "cuda"
  num_workers: 4

model:
  architecture: "resnet50"
  pretrained: true

optimizer:
  name: "AdamW"
  lr: 0.001
  weight_decay: 0.01
```

### 2. 训练管理

#### 启动训练
```bash
# 在AutoDL上
tmux new-session -d -s training 'python src/03_cnn/fruits_classifier.py --config configs/autodl.yaml'
```

#### 查看训练进度
```bash
# 连接到训练会话
tmux attach -t training

# 查看日志
tail -f outputs/logs/training.log

# 监控GPU使用
nvidia-smi
```

#### 后台运行
```bash
# 分离会话
Ctrl+B, D

# 列出会话
tmux list-sessions

# 重新连接
tmux attach -t training
```

### 3. 结果管理

#### 模型检查点
```
outputs/
├── checkpoints/
│   ├── best_model.pth
│   ├── epoch_10.pth
│   └── latest.pth
├── logs/
│   ├── training.log
│   └── tensorboard/
└── results/
    ├── metrics.json
    └── confusion_matrix.png
```

#### 下载结果
```bash
# 从AutoDL下载训练结果
scp -r root@your-instance:/root/ai-learning/outputs/ ./outputs/
```

## 📊 监控和调试

### 1. 实时监控

#### 训练状态
```bash
# 本地监控
python scripts/sync_to_autodl.py --action monitor

# 远程监控
ssh root@your-instance "tail -f /root/ai-learning/outputs/logs/training.log"
```

#### 系统资源
```bash
# GPU监控
watch -n 1 nvidia-smi

# 内存监控
htop

# 磁盘空间
df -h
```

### 2. 调试技巧

#### 本地调试
```bash
# 使用调试配置
python scripts/local_dev.py --action config --config-template debug
python src/03_cnn/fruits_classifier.py --config configs/debug.yaml
```

#### 远程调试
```bash
# SSH端口转发用于Jupyter
ssh -L 8888:localhost:8888 root@your-instance

# 启动Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 3. 日志分析

#### 训练日志
```bash
# 查看最新日志
tail -n 50 outputs/logs/training.log

# 搜索错误
grep -i error outputs/logs/training.log

# 查看训练指标
grep "Epoch\|Loss\|Accuracy" outputs/logs/training.log
```

## 💡 最佳实践

### 1. 开发流程
1. **本地快速验证**: 使用样本数据和CPU配置
2. **代码质量检查**: 运行lint和测试
3. **配置优化**: 针对不同环境调整参数
4. **自动化部署**: 使用同步脚本一键部署
5. **监控训练**: 实时跟踪训练进度

### 2. 配置管理
- 使用不同配置文件区分环境
- 版本控制配置文件
- 参数化关键设置

### 3. 数据管理
- 本地使用样本数据
- 云端使用完整数据集
- 定期备份重要结果

### 4. 代码组织
```
src/
├── models/          # 模型定义
├── utils/           # 工具函数
├── data/            # 数据处理
└── training/        # 训练脚本

configs/             # 配置文件
scripts/             # 自动化脚本
docs/                # 文档
requirements/        # 依赖管理
```

## 🔧 故障排除

### 1. 常见问题

#### SSH连接失败
```bash
# 检查SSH配置
ssh -v root@your-instance

# 重新生成密钥
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_autodl
```

#### 依赖安装失败
```bash
# 清理pip缓存
pip cache purge

# 使用国内镜像
pip install -r requirements/autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 训练中断
```bash
# 检查tmux会话
tmux list-sessions

# 查看系统日志
journalctl -f

# 检查磁盘空间
df -h
```

### 2. 性能优化

#### 数据加载优化
```python
# 增加数据加载进程
num_workers = 4

# 使用内存映射
pin_memory = True

# 预取数据
prefetch_factor = 2
```

#### GPU优化
```python
# 混合精度训练
torch.cuda.amp.autocast()

# 梯度累积
accumulation_steps = 4

# 模型并行
torch.nn.DataParallel()
```

### 3. 调试工具

#### 性能分析
```python
# PyTorch Profiler
with torch.profiler.profile() as prof:
    model(input)
print(prof.key_averages().table())
```

#### 内存监控
```python
# GPU内存监控
torch.cuda.memory_summary()

# 内存泄漏检测
import tracemalloc
tracemalloc.start()
```

## 📚 参考资源

### 文档
- [PyTorch官方文档](https://pytorch.org/docs/)
- [AutoDL使用指南](https://www.autodl.com/docs/)
- [Tmux使用教程](https://github.com/tmux/tmux/wiki)

### 工具
- [Weights & Biases](https://wandb.ai/) - 实验跟踪
- [TensorBoard](https://www.tensorflow.org/tensorboard) - 可视化
- [Hydra](https://hydra.cc/) - 配置管理

### 社区
- [PyTorch论坛](https://discuss.pytorch.org/)
- [AutoDL社区](https://www.autodl.com/community/)

---

## 🎉 快速开始示例

```bash
# 1. 设置本地环境
python scripts/local_dev.py --action setup

# 2. 创建样本数据
python scripts/local_dev.py --action sample

# 3. 本地测试
python scripts/local_dev.py --action test

# 4. 配置AutoDL连接
# 编辑 configs/sync_config.yaml

# 5. 一键部署和训练
python scripts/sync_to_autodl.py --action full

# 6. 监控训练
python scripts/sync_to_autodl.py --action monitor
```

现在你已经拥有了一个完整的Trae + AutoDL开发训练工作流！🚀