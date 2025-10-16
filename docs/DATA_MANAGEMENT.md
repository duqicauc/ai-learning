# 数据管理指南

## 📊 数据目录结构

```
ai-learning/
├── data/                    # 数据根目录（符号链接到外部数据）
│   ├── cats_and_dogs/      # 猫狗分类数据集
│   │   ├── train/          # 训练集
│   │   │   ├── cat/        # 猫的图片
│   │   │   └── dog/        # 狗的图片
│   │   ├── val/            # 验证集
│   │   │   ├── cat/
│   │   │   └── dog/
│   │   ├── README.md       # 数据集说明
│   │   ├── classname.txt   # 类别名称
│   │   └── *.csv           # 数据集元信息
│   ├── cifar10/            # CIFAR-10数据集
│   ├── mnist/              # MNIST数据集
│   └── custom/             # 自定义数据集
└── scripts/
    ├── download_datasets.py    # 数据集下载脚本
    ├── validate_data.py        # 数据验证脚本
    └── preprocess_data.py      # 数据预处理脚本
```

## 🎯 数据管理原则

### 1. 数据分离
- **原则**: 数据与代码分离，避免将大文件提交到Git
- **实现**: 使用符号链接或相对路径引用外部数据目录
- **好处**: 减小仓库大小，提高克隆速度

### 2. 标准化结构
- **训练集**: `train/` - 用于模型训练
- **验证集**: `val/` - 用于超参数调优和模型选择
- **测试集**: `test/` - 用于最终性能评估（如果有）
- **元数据**: README.md, classname.txt, *.csv

### 3. 版本控制
- **数据版本**: 使用日期或版本号标识数据集版本
- **变更记录**: 在README.md中记录数据集变更历史
- **备份策略**: 重要数据集保持多个备份

## 🔧 数据配置管理

### 统一配置文件
使用 `src/utils/data_config.py` 统一管理所有数据路径：

```python
from src.utils.data_config import get_cats_dogs_paths

# 获取数据路径
train_dir, val_dir, test_dir = get_cats_dogs_paths()

# 验证数据集
from src.utils.data_config import validate_all_datasets
validate_all_datasets()
```

### 环境适配
- **开发环境**: 使用相对路径，便于团队协作
- **生产环境**: 支持绝对路径配置
- **云环境**: 支持云存储路径配置

## 📥 数据集获取方式

### 1. 公开数据集
```bash
# 使用脚本下载
python scripts/download_datasets.py --dataset cats_and_dogs

# 手动下载
# 1. 访问数据集官网
# 2. 下载到 data/ 目录
# 3. 解压并重命名为标准格式
```

### 2. 自定义数据集
```bash
# 创建数据集目录
mkdir -p data/my_dataset/{train,val,test}

# 按类别组织数据
data/my_dataset/
├── train/
│   ├── class1/
│   └── class2/
├── val/
│   ├── class1/
│   └── class2/
└── README.md
```

## 🔍 数据验证

### 自动验证
```python
# 验证数据集完整性
python -m src.utils.data_config

# 验证特定数据集
from src.utils.data_config import data_config
data_config.validate_dataset("cats_and_dogs")
```

### 手动检查清单
- [ ] 目录结构正确
- [ ] 类别目录存在
- [ ] 图片文件可正常读取
- [ ] 类别分布合理
- [ ] 文件命名规范

## 📈 数据统计和分析

### 基本统计
```python
# 数据集统计脚本
python scripts/analyze_dataset.py --dataset cats_and_dogs

# 输出示例：
# 📊 数据集统计: cats_and_dogs
# 总样本数: 345
# 训练集: 275 (79.7%)
# 验证集: 70 (20.3%)
# 类别分布:
#   - cat: 172 (49.9%)
#   - dog: 173 (50.1%)
```

### 数据质量检查
- **图片尺寸分布**: 检查是否需要统一尺寸
- **文件格式**: 确保格式一致性
- **损坏文件**: 检测无法读取的文件
- **类别平衡**: 分析类别分布是否均衡

## 🚀 数据预处理

### 图像预处理管道
```python
from torchvision import transforms

# 训练时数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 验证/测试时预处理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 数据增强策略
- **几何变换**: 旋转、翻转、缩放、裁剪
- **颜色变换**: 亮度、对比度、饱和度调整
- **噪声添加**: 高斯噪声、椒盐噪声
- **混合技术**: Mixup, CutMix

## 💾 数据存储最佳实践

### 文件组织
```bash
# 推荐的文件命名
train/cat/cat_001.jpg
train/cat/cat_002.jpg
train/dog/dog_001.jpg

# 避免的命名方式
train/cat/IMG_20240101_001.jpg  # 无意义命名
train/cat/猫咪照片1.jpg         # 中文命名
```

### 存储优化
- **图片格式**: 使用JPEG压缩减小文件大小
- **分辨率**: 根据模型需求选择合适分辨率
- **批量处理**: 使用脚本批量转换格式和尺寸

## 🔒 数据安全

### 访问控制
- **敏感数据**: 使用加密存储
- **访问权限**: 限制数据访问权限
- **备份策略**: 定期备份重要数据

### 隐私保护
- **数据脱敏**: 移除个人身份信息
- **使用协议**: 遵守数据使用协议
- **合规检查**: 确保符合相关法规

## 🛠️ 常用工具和脚本

### 数据集管理工具
```bash
# 下载数据集
python scripts/download_datasets.py

# 验证数据集
python scripts/validate_data.py

# 数据集统计
python scripts/analyze_dataset.py

# 数据预处理
python scripts/preprocess_data.py
```

### 第三方工具
- **DVC**: 数据版本控制
- **MLflow**: 实验和数据追踪
- **Weights & Biases**: 数据集可视化
- **Roboflow**: 数据标注和管理

## 📋 数据集清单

### 当前可用数据集
| 数据集名称 | 类别数 | 训练样本 | 验证样本 | 用途 | 状态 |
|-----------|--------|----------|----------|------|------|
| cats_and_dogs | 2 | 275 | 70 | 图像分类 | ✅ 可用 |
| cifar10 | 10 | 50000 | 10000 | 图像分类 | 🔄 待下载 |
| mnist | 10 | 60000 | 10000 | 手写数字识别 | 🔄 待下载 |

### 计划添加的数据集
- [ ] ImageNet子集
- [ ] COCO数据集
- [ ] 自定义业务数据集

## 🚨 常见问题和解决方案

### 问题1: 数据路径错误
**症状**: FileNotFoundError: 找不到数据文件
**解决**: 
1. 检查数据配置文件路径
2. 验证符号链接是否正确
3. 确认数据目录结构

### 问题2: 内存不足
**症状**: CUDA out of memory
**解决**:
1. 减小batch_size
2. 使用数据流式加载
3. 优化图片分辨率

### 问题3: 数据加载慢
**症状**: 训练时数据加载成为瓶颈
**解决**:
1. 增加num_workers
2. 使用SSD存储
3. 预处理数据缓存

## 📚 参考资源

- [PyTorch数据加载教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [数据增强最佳实践](https://github.com/aleju/imgaug)
- [DVC数据版本控制](https://dvc.org/)
- [MLOps数据管理](https://ml-ops.org/)