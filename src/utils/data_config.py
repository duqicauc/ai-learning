"""
数据配置管理模块 - 简化版

统一管理项目中所有数据集的路径配置，支持：
- 双环境配置（本地开发、AutoDL云端）
- 自动环境检测和路径切换
- 数据集路径验证
- 数据集信息查询
- 小样本测试数据集
"""

import os
import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from enum import Enum

class DataEnvironment(Enum):
    """数据环境类型"""
    LOCAL = "local"          # 本地开发环境
    AUTODL = "autodl"        # AutoDL云端环境

class DataConfig:
    """数据配置管理类 - 简化版"""
    
    def __init__(self, force_env: Optional[str] = None):
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        
        # 检测当前环境
        self.current_env = self._detect_environment() if not force_env else DataEnvironment(force_env)
        
        # 根据环境设置数据根目录
        self.data_root = self._get_data_root()
        
        # 数据集配置
        self.datasets = self._init_datasets()
        
        print(f"🌍 当前数据环境: {self.current_env.value}")
        print(f"📁 数据根目录: {self.data_root}")
    
    def _detect_environment(self) -> DataEnvironment:
        """自动检测当前环境"""
        # 检查是否在AutoDL环境
        if os.path.exists("/root/miniconda3") or os.path.exists("/autodl-tmp") or os.path.exists("/root/autodl-fs"):
            return DataEnvironment.AUTODL
        
        # 默认为本地环境
        return DataEnvironment.LOCAL
    
    def _get_data_root(self) -> Path:
        """根据环境获取数据根目录"""
        if self.current_env == DataEnvironment.LOCAL:
            return self.project_root / "data"
        
        elif self.current_env == DataEnvironment.AUTODL:
            # AutoDL环境优先级：文件存储 -> 项目data目录 -> /root/data -> /autodl-tmp
            autodl_paths = [
                Path("/root/autodl-fs/data"),      # AutoDL文件存储挂载目录
                Path("/root/autodl-fs"),           # AutoDL文件存储根目录
                self.project_root / "data",        # 项目data目录
                Path("/root/data"),                # 用户根目录data
                Path("/autodl-tmp/data"),          # 临时存储data
                Path("/autodl-pub")                # 公共数据集目录
            ]
            
            for path in autodl_paths:
                if path.exists():
                    return path
            
            # 如果都不存在，使用项目data目录
            return self.project_root / "data"
        
        return self.project_root / "data"
    
    def _init_datasets(self) -> Dict:
        """初始化数据集配置"""
        base_config = {
            "cats_and_dogs": {
                "train_dir": "train",
                "val_dir": "val", 
                "test_dir": "val",
                "classes": ["cat", "dog"],
                "description": "猫狗二分类数据集，275张训练图片，70张验证图片",
                "small_sample_size": 20  # 小样本测试数量
            },
            "fruits100": {
                "train_dir": "train",
                "val_dir": "val",
                "test_dir": "val",
                "num_classes": 100,
                "description": "水果分类数据集，包含100种水果类型的图像，支持多分类任务",
                "source": "https://www.modelscope.cn/datasets/tany0699/fruits100",
                "format": "JPG图片，按文件夹分类组织",
                "small_sample_size": 50  # 小样本测试数量
            },
            "cifar10": {
                "classes": ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"],
                "description": "CIFAR-10数据集，10个类别的32x32彩色图片",
                "small_sample_size": 100
            }
        }
        
        # 根据环境设置路径
        datasets = {}
        for name, config in base_config.items():
            datasets[name] = config.copy()
            datasets[name]["paths"] = self._get_dataset_paths(name)
        
        return datasets
    
    def _get_dataset_paths(self, dataset_name: str) -> Dict[str, Path]:
        """获取数据集在不同环境下的路径"""
        paths = {}
        
        if self.current_env == DataEnvironment.LOCAL:
            paths["main"] = self.data_root / dataset_name
            paths["small"] = self.data_root / f"{dataset_name}_small"
        
        elif self.current_env == DataEnvironment.AUTODL:
             # AutoDL环境路径优先级
             autodl_candidates = [
                 self.data_root / dataset_name,
                 Path(f"/root/autodl-fs/data/{dataset_name}"),    # 文件存储data目录
                 Path(f"/root/autodl-fs/{dataset_name}"),         # 文件存储根目录
                 Path(f"/root/data/{dataset_name}"),              # 用户根目录
                 Path(f"/autodl-tmp/{dataset_name}"),             # 临时存储
                 Path(f"/autodl-pub/{dataset_name}")              # 公共数据集
             ]
             
             # 找到第一个存在的路径
             main_path = None
             for candidate in autodl_candidates:
                 if candidate.exists():
                     main_path = candidate
                     break
             
             paths["main"] = main_path or autodl_candidates[0]
             paths["small"] = self.data_root / f"{dataset_name}_small"
        
        return paths
    
    def get_dataset_path(self, dataset_name: str, use_small: bool = False) -> Path:
        """获取数据集路径"""
        if dataset_name not in self.datasets:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        paths = self.datasets[dataset_name]["paths"]
        
        if use_small:
            dataset_path = paths["small"]
        else:
            dataset_path = paths["main"]
        
        if not dataset_path.exists() and not use_small:
            # 如果主数据集不存在，尝试创建小样本数据集
            print(f"⚠️ 主数据集不存在: {dataset_path}")
            print(f"🔄 尝试创建小样本数据集...")
            if self.create_small_dataset(dataset_name):
                return paths["small"]
            else:
                raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        return dataset_path
    
    def get_cats_dogs_paths(self, use_small: bool = False) -> Tuple[Path, Path, Path]:
        """获取猫狗数据集的训练、验证、测试路径"""
        dataset_path = self.get_dataset_path("cats_and_dogs", use_small)
        config = self.datasets["cats_and_dogs"]
        
        train_path = dataset_path / config["train_dir"]
        val_path = dataset_path / config["val_dir"]
        test_path = dataset_path / config["test_dir"]
        
        return train_path, val_path, test_path
    
    def get_fruits100_paths(self, use_small: bool = False) -> Tuple[Path, Path, Path]:
        """获取水果数据集的训练、验证、测试路径"""
        dataset_path = self.get_dataset_path("fruits100", use_small)
        config = self.datasets["fruits100"]
        
        train_path = dataset_path / config["train_dir"]
        val_path = dataset_path / config["val_dir"]
        test_path = dataset_path / config["test_dir"]
        
        return train_path, val_path, test_path
    
    def create_small_dataset(self, dataset_name: str) -> bool:
        """创建小样本测试数据集"""
        try:
            if dataset_name not in self.datasets:
                return False
            
            config = self.datasets[dataset_name]
            small_path = config["paths"]["small"]
            sample_size = config.get("small_sample_size", 20)
            
            print(f"📁 创建小样本数据集: {small_path}")
            
            if dataset_name == "cats_and_dogs":
                return self._create_cats_dogs_small(small_path, sample_size)
            elif dataset_name == "fruits100":
                return self._create_fruits100_small(small_path, sample_size)
            
            return False
            
        except Exception as e:
            print(f"❌ 创建小样本数据集失败: {e}")
            return False
    
    def _create_cats_dogs_small(self, small_path: Path, sample_size: int) -> bool:
        """创建猫狗小样本数据集"""
        try:
            # 创建目录结构
            for split in ["train", "val"]:
                for class_name in ["cat", "dog"]:
                    class_dir = small_path / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 创建占位符文件
                    for i in range(sample_size // 4):  # 每个类别每个split创建几个文件
                        placeholder = class_dir / f"{class_name}_{i:03d}.jpg"
                        placeholder.touch()
            
            print(f"✅ 猫狗小样本数据集创建完成: {small_path}")
            return True
            
        except Exception as e:
            print(f"❌ 创建猫狗小样本数据集失败: {e}")
            return False
    
    def _create_fruits100_small(self, small_path: Path, sample_size: int) -> bool:
        """创建水果小样本数据集"""
        try:
            # 常见水果类别
            fruit_classes = [
                "apple", "banana", "orange", "grape", "strawberry",
                "pineapple", "mango", "watermelon", "peach", "pear"
            ]
            
            # 创建目录结构
            for split in ["train", "val"]:
                for fruit in fruit_classes:
                    class_dir = small_path / split / fruit
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 创建占位符文件
                    files_per_class = sample_size // len(fruit_classes) // 2  # 分配到train和val
                    for i in range(max(1, files_per_class)):
                        placeholder = class_dir / f"{fruit}_{i:03d}.jpg"
                        placeholder.touch()
            
            print(f"✅ 水果小样本数据集创建完成: {small_path}")
            return True
            
        except Exception as e:
            print(f"❌ 创建水果小样本数据集失败: {e}")
            return False
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.datasets:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        info = self.datasets[dataset_name].copy()
        info["current_env"] = self.current_env.value
        info["available_paths"] = {
            "main": str(info["paths"]["main"]),
            "small": str(info["paths"]["small"])
        }
        
        return info
    
    def list_datasets(self) -> Dict[str, str]:
        """列出所有可用数据集"""
        return {name: config["description"] 
                for name, config in self.datasets.items()}
    
    def validate_dataset(self, dataset_name: str, use_small: bool = False) -> bool:
        """验证数据集是否存在且结构正确"""
        try:
            dataset_path = self.get_dataset_path(dataset_name, use_small)
            
            if dataset_name == "cats_and_dogs":
                train_path, val_path, _ = self.get_cats_dogs_paths(use_small)
                
                # 检查目录结构
                required_dirs = [train_path, val_path]
                for dir_path in required_dirs:
                    if not dir_path.exists():
                        print(f"❌ 缺少目录: {dir_path}")
                        return False
                
                # 检查类别目录
                for split_dir in [train_path, val_path]:
                    for class_name in self.datasets[dataset_name]["classes"]:
                        class_dir = split_dir / class_name
                        if not class_dir.exists():
                            print(f"❌ 缺少类别目录: {class_dir}")
                            return False
            
            elif dataset_name == "fruits100":
                train_path, val_path, _ = self.get_fruits100_paths(use_small)
                
                # 检查目录结构
                required_dirs = [train_path, val_path]
                for dir_path in required_dirs:
                    if not dir_path.exists():
                        print(f"❌ 缺少目录: {dir_path}")
                        return False
                
                # 检查是否有子目录（类别目录）
                train_subdirs = [d for d in train_path.iterdir() if d.is_dir()]
                val_subdirs = [d for d in val_path.iterdir() if d.is_dir()]
                
                if len(train_subdirs) == 0:
                    print(f"❌ 训练集目录为空: {train_path}")
                    return False
                
                if len(val_subdirs) == 0:
                    print(f"❌ 验证集目录为空: {val_path}")
                    return False
                
                print(f"✅ 发现 {len(train_subdirs)} 个训练类别，{len(val_subdirs)} 个验证类别")
            
            print(f"✅ 数据集 {dataset_name} 验证通过")
            return True
                
        except Exception as e:
            print(f"❌ 数据集验证失败: {e}")
            return False
    
    def switch_environment(self, env: str) -> bool:
        """切换数据环境"""
        try:
            new_env = DataEnvironment(env)
            old_env = self.current_env
            
            self.current_env = new_env
            self.data_root = self._get_data_root()
            self.datasets = self._init_datasets()
            
            print(f"🔄 环境切换: {old_env.value} -> {new_env.value}")
            print(f"📁 新数据根目录: {self.data_root}")
            
            return True
            
        except ValueError as e:
            print(f"❌ 环境切换失败: {e}")
            return False
    
    def get_relative_path(self, dataset_name: str, use_small: bool = False) -> str:
        """获取相对于当前工作目录的数据集路径"""
        dataset_path = self.get_dataset_path(dataset_name, use_small)
        try:
            relative_path = os.path.relpath(dataset_path)
            return relative_path
        except ValueError:
            return str(dataset_path)

# 全局数据配置实例
data_config = DataConfig()

# 便捷函数
def get_cats_dogs_paths(use_small: bool = False):
    """获取猫狗数据集路径的便捷函数"""
    return data_config.get_cats_dogs_paths(use_small)

def get_fruits100_paths(use_small: bool = False):
    """获取水果数据集路径的便捷函数"""
    return data_config.get_fruits100_paths(use_small)

def get_dataset_path(dataset_name: str, use_small: bool = False):
    """获取数据集路径的便捷函数"""
    return data_config.get_dataset_path(dataset_name, use_small)

def validate_all_datasets(use_small: bool = False):
    """验证所有数据集"""
    print("🔍 验证数据集...")
    for dataset_name in data_config.datasets.keys():
        data_config.validate_dataset(dataset_name, use_small)

def create_small_datasets():
    """创建所有小样本数据集"""
    print("📁 创建小样本数据集...")
    for dataset_name in data_config.datasets.keys():
        data_config.create_small_dataset(dataset_name)

if __name__ == "__main__":
    # 测试数据配置
    print("📊 数据配置测试 - 重构版")
    print("=" * 50)
    
    # 显示环境信息
    print(f"\n🌍 当前环境: {data_config.current_env.value}")
    print(f"📁 数据根目录: {data_config.data_root}")
    
    # 列出所有数据集
    print("\n📋 可用数据集:")
    for name, desc in data_config.list_datasets().items():
        print(f"  • {name}: {desc}")
    
    # 创建小样本数据集
    print("\n📁 创建小样本数据集:")
    create_small_datasets()
    
    # 验证数据集
    print("\n🔍 验证数据集:")
    validate_all_datasets(use_small=True)
    
    # 显示路径信息
    print("\n📁 路径信息:")
    try:
        train_path, val_path, test_path = get_fruits100_paths(use_small=True)
        print(f"  训练集: {train_path}")
        print(f"  验证集: {val_path}")
        print(f"  测试集: {test_path}")
    except Exception as e:
        print(f"  ❌ 获取路径失败: {e}")