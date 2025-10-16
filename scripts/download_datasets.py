#!/usr/bin/env python3
"""
数据集下载和管理脚本

支持下载常用的机器学习数据集，并自动组织为标准目录结构。

使用方法:
    python scripts/download_datasets.py --dataset cats_and_dogs
    python scripts/download_datasets.py --dataset cifar10
    python scripts/download_datasets.py --list
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 数据集配置
DATASETS_CONFIG = {
    "cats_and_dogs": {
        "name": "Cats and Dogs Classification",
        "description": "二分类数据集，包含猫和狗的图片",
        "url": "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
        "size": "~800MB",
        "classes": ["cat", "dog"],
        "format": "jpg",
        "structure": "PetImages/{Cat,Dog}/*.jpg",
        "preprocessing": "需要清理损坏的图片文件"
    },
    "fruits100": {
        "name": "Fruits 100 Classification",
        "description": "100种水果分类数据集，包含丰富的水果图像",
        "url": "https://www.modelscope.cn/datasets/tany0699/fruits100.git",
        "size": "~500MB",
        "classes": "100种水果类别",
        "format": "jpg",
        "structure": "train/{fruit_name}/*.jpg, val/{fruit_name}/*.jpg",
        "preprocessing": "ModelScope数据集，需要使用git clone下载",
        "download_method": "git_clone"
    },
    "cifar10": {
        "name": "CIFAR-10",
        "description": "10类物体识别数据集",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "size": "~170MB",
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "format": "pickle",
        "structure": "cifar-10-batches-py/",
        "preprocessing": "需要转换为图片格式"
    },
    "mnist": {
        "name": "MNIST",
        "description": "手写数字识别数据集",
        "url": "http://yann.lecun.com/exdb/mnist/",
        "size": "~60MB",
        "classes": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "format": "idx",
        "structure": "MNIST/raw/",
        "preprocessing": "PyTorch可直接加载"
    }
}

class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
    def list_datasets(self) -> None:
        """列出所有可用的数据集"""
        print("📊 可用数据集列表:")
        print("=" * 60)
        
        for dataset_id, config in DATASETS_CONFIG.items():
            status = "✅ 已下载" if self._is_downloaded(dataset_id) else "⬇️ 未下载"
            print(f"\n🔹 {dataset_id}")
            print(f"   名称: {config['name']}")
            print(f"   描述: {config['description']}")
            print(f"   大小: {config['size']}")
            print(f"   类别数: {len(config['classes'])}")
            print(f"   状态: {status}")
    
    def _is_downloaded(self, dataset_id: str) -> bool:
        """检查数据集是否已下载"""
        dataset_path = self.data_root / dataset_id
        return dataset_path.exists() and any(dataset_path.iterdir())
    
    def download_dataset(self, dataset_id: str, force: bool = False) -> bool:
        """
        下载指定数据集
        
        Args:
            dataset_id: 数据集ID
            force: 是否强制重新下载
            
        Returns:
            bool: 下载是否成功
        """
        if dataset_id not in DATASETS_CONFIG:
            print(f"❌ 未知数据集: {dataset_id}")
            print(f"可用数据集: {list(DATASETS_CONFIG.keys())}")
            return False
        
        config = DATASETS_CONFIG[dataset_id]
        dataset_path = self.data_root / dataset_id
        
        # 检查是否已存在
        if self._is_downloaded(dataset_id) and not force:
            print(f"✅ 数据集 {dataset_id} 已存在，跳过下载")
            print(f"   路径: {dataset_path}")
            print(f"   使用 --force 强制重新下载")
            return True
        
        print(f"📥 开始下载数据集: {config['name']}")
        print(f"   大小: {config['size']}")
        print(f"   URL: {config['url']}")
        
        try:
            # 创建数据集目录
            dataset_path.mkdir(exist_ok=True)
            
            # 根据数据集类型调用相应的下载方法
            if dataset_id == "cats_and_dogs":
                return self._download_cats_and_dogs(config, dataset_path)
            elif dataset_id == "cifar10":
                return self._download_cifar10(config, dataset_path)
            elif dataset_id == "mnist":
                return self._download_mnist(config, dataset_path)
            else:
                print(f"❌ 暂不支持自动下载 {dataset_id}")
                return False
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    def _download_with_progress(self, url: str, filepath: Path) -> None:
        """带进度条的文件下载"""
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r   下载进度: {percent:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
        
        print(f"   正在下载到: {filepath}")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # 换行
    
    def _download_cats_and_dogs(self, config: Dict, dataset_path: Path) -> bool:
        """下载猫狗数据集"""
        zip_path = dataset_path / "kagglecatsanddogs.zip"
        
        # 下载ZIP文件
        self._download_with_progress(config["url"], zip_path)
        
        print("   正在解压文件...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # 重组目录结构
        print("   正在重组目录结构...")
        pet_images_path = dataset_path / "PetImages"
        
        if pet_images_path.exists():
            # 创建标准目录结构
            train_path = dataset_path / "train"
            val_path = dataset_path / "val"
            train_path.mkdir(exist_ok=True)
            val_path.mkdir(exist_ok=True)
            
            for class_name in ["Cat", "Dog"]:
                class_path = pet_images_path / class_name
                if class_path.exists():
                    # 创建类别目录
                    train_class_path = train_path / class_name.lower()
                    val_class_path = val_path / class_name.lower()
                    train_class_path.mkdir(exist_ok=True)
                    val_class_path.mkdir(exist_ok=True)
                    
                    # 获取所有图片文件
                    image_files = list(class_path.glob("*.jpg"))
                    print(f"   处理 {class_name}: {len(image_files)} 张图片")
                    
                    # 清理损坏的文件
                    valid_files = []
                    for img_file in image_files:
                        try:
                            # 简单检查文件大小
                            if img_file.stat().st_size > 1000:  # 大于1KB
                                valid_files.append(img_file)
                        except:
                            continue
                    
                    print(f"   有效图片: {len(valid_files)} 张")
                    
                    # 分割训练集和验证集 (80:20)
                    split_idx = int(len(valid_files) * 0.8)
                    train_files = valid_files[:split_idx]
                    val_files = valid_files[split_idx:]
                    
                    # 复制文件
                    for i, img_file in enumerate(train_files):
                        new_name = f"{class_name.lower()}_{i+1:04d}.jpg"
                        shutil.copy2(img_file, train_class_path / new_name)
                    
                    for i, img_file in enumerate(val_files):
                        new_name = f"{class_name.lower()}_{i+1:04d}.jpg"
                        shutil.copy2(img_file, val_class_path / new_name)
            
            # 清理原始文件
            shutil.rmtree(pet_images_path)
        
        # 删除ZIP文件
        zip_path.unlink()
        
        # 创建README文件
        self._create_dataset_readme(dataset_path, config)
        
        print("✅ 猫狗数据集下载完成!")
        return True
    
    def _download_cifar10(self, config: Dict, dataset_path: Path) -> bool:
        """下载CIFAR-10数据集"""
        print("ℹ️  CIFAR-10数据集建议使用PyTorch内置下载:")
        print("   from torchvision import datasets")
        print("   datasets.CIFAR10(root='data', train=True, download=True)")
        
        # 这里可以实现自定义下载逻辑
        return False
    
    def _download_mnist(self, config: Dict, dataset_path: Path) -> bool:
        """下载MNIST数据集"""
        print("ℹ️  MNIST数据集建议使用PyTorch内置下载:")
        print("   from torchvision import datasets")
        print("   datasets.MNIST(root='data', train=True, download=True)")
        
        # 这里可以实现自定义下载逻辑
        return False
    
    def _create_dataset_readme(self, dataset_path: Path, config: Dict) -> None:
        """创建数据集README文件"""
        readme_content = f"""# {config['name']}

## 数据集信息
- **描述**: {config['description']}
- **大小**: {config['size']}
- **类别数**: {len(config['classes'])}
- **格式**: {config['format']}

## 类别列表
{chr(10).join(f"- {cls}" for cls in config['classes'])}

## 目录结构
```
{dataset_path.name}/
├── train/          # 训练集
│   ├── {config['classes'][0]}/
│   └── {config['classes'][1] if len(config['classes']) > 1 else '...'}/
├── val/            # 验证集
│   ├── {config['classes'][0]}/
│   └── {config['classes'][1] if len(config['classes']) > 1 else '...'}/
└── README.md       # 本文件
```

## 使用方法
```python
from src.utils.data_config import get_{dataset_path.name.replace('-', '_')}_paths

# 获取数据路径
train_dir, val_dir, test_dir = get_{dataset_path.name.replace('-', '_')}_paths()

# 加载数据集
from torchvision import datasets, transforms
train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
```

## 数据统计
- 训练集: 待统计
- 验证集: 待统计
- 总计: 待统计

## 下载时间
{Path(__file__).stat().st_mtime}

## 预处理说明
{config.get('preprocessing', '无特殊预处理需求')}
"""
        
        readme_path = dataset_path / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集下载和管理工具")
    parser.add_argument("--dataset", type=str, help="要下载的数据集名称")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_root)
    
    if args.list:
        downloader.list_datasets()
    elif args.dataset:
        success = downloader.download_dataset(args.dataset, args.force)
        if success:
            print(f"\n🎉 数据集 {args.dataset} 准备就绪!")
            print(f"   路径: {Path(args.data_root) / args.dataset}")
        else:
            print(f"\n❌ 数据集 {args.dataset} 下载失败")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()