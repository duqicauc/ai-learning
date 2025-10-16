"""
数据配置管理模块

统一管理项目中所有数据集的路径配置，支持：
- 相对路径和绝对路径自动切换
- 多环境配置（开发、生产）
- 数据集路径验证
- 数据集信息查询
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

class DataConfig:
    """数据配置管理类"""
    
    def __init__(self):
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        
        # 数据根目录（相对于项目根目录）
        self.data_root = self.project_root / "data"
        
        # 数据集配置
        self.datasets = {
            "cats_and_dogs": {
                "path": self.data_root / "cats_and_dogs",
                "train_dir": "train",
                "val_dir": "val", 
                "test_dir": "val",  # 使用验证集作为测试集
                "classes": ["cat", "dog"],
                "description": "猫狗二分类数据集，275张训练图片，70张验证图片"
            },
            "fruits100": {
                "path": self.data_root / "fruits100",
                "train_dir": "train",
                "val_dir": "val",
                "test_dir": "val",  # 使用验证集作为测试集
                "num_classes": 100,
                "description": "水果分类数据集，包含100种水果类型的图像，支持多分类任务",
                "source": "https://www.modelscope.cn/datasets/tany0699/fruits100",
                "format": "JPG图片，按文件夹分类组织"
            },
            "cifar10": {
                "path": self.data_root / "cifar-10-python.tar.gz",
                "classes": ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"],
                "description": "CIFAR-10数据集，10个类别的32x32彩色图片"
            }
        }
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """获取数据集路径"""
        if dataset_name not in self.datasets:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        dataset_path = self.datasets[dataset_name]["path"]
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        return dataset_path
    
    def get_cats_dogs_paths(self) -> Tuple[Path, Path, Path]:
        """获取猫狗数据集的训练、验证、测试路径"""
        dataset_path = self.get_dataset_path("cats_and_dogs")
        config = self.datasets["cats_and_dogs"]
        
        train_path = dataset_path / config["train_dir"]
        val_path = dataset_path / config["val_dir"]
        test_path = dataset_path / config["test_dir"]
        
        return train_path, val_path, test_path
    
    def get_fruits100_paths(self) -> Tuple[Path, Path, Path]:
        """获取水果数据集的训练、验证、测试路径"""
        dataset_path = self.get_dataset_path("fruits100")
        config = self.datasets["fruits100"]
        
        train_path = dataset_path / config["train_dir"]
        val_path = dataset_path / config["val_dir"]
        test_path = dataset_path / config["test_dir"]
        
        return train_path, val_path, test_path
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name not in self.datasets:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        return self.datasets[dataset_name].copy()
    
    def list_datasets(self) -> Dict[str, str]:
        """列出所有可用数据集"""
        return {name: config["description"] 
                for name, config in self.datasets.items()}
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """验证数据集是否存在且结构正确"""
        try:
            dataset_path = self.get_dataset_path(dataset_name)
            
            if dataset_name == "cats_and_dogs":
                train_path, val_path, _ = self.get_cats_dogs_paths()
                
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
                train_path, val_path, _ = self.get_fruits100_paths()
                
                # 检查目录结构
                required_dirs = [train_path, val_path]
                for dir_path in required_dirs:
                    if not dir_path.exists():
                        print(f"❌ 缺少目录: {dir_path}")
                        return False
                
                # 对于水果数据集，检查是否有子目录（类别目录）
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
    
    def get_relative_path(self, dataset_name: str) -> str:
        """获取相对于当前工作目录的数据集路径"""
        dataset_path = self.get_dataset_path(dataset_name)
        try:
            # 尝试获取相对路径
            relative_path = os.path.relpath(dataset_path)
            return relative_path
        except ValueError:
            # 如果无法获取相对路径，返回绝对路径
            return str(dataset_path)

# 全局数据配置实例
data_config = DataConfig()

# 便捷函数
def get_cats_dogs_paths():
    """获取猫狗数据集路径的便捷函数"""
    return data_config.get_cats_dogs_paths()

def get_fruits100_paths():
    """获取水果数据集路径的便捷函数"""
    return data_config.get_fruits100_paths()

def get_dataset_path(dataset_name: str):
    """获取数据集路径的便捷函数"""
    return data_config.get_dataset_path(dataset_name)

def validate_all_datasets():
    """验证所有数据集"""
    print("🔍 验证数据集...")
    for dataset_name in data_config.datasets.keys():
        data_config.validate_dataset(dataset_name)

if __name__ == "__main__":
    # 测试数据配置
    print("📊 数据配置测试")
    print("=" * 50)
    
    # 列出所有数据集
    print("\n📋 可用数据集:")
    for name, desc in data_config.list_datasets().items():
        print(f"  • {name}: {desc}")
    
    # 验证数据集
    print("\n🔍 验证数据集:")
    validate_all_datasets()
    
    # 显示路径信息
    print("\n📁 路径信息:")
    try:
        train_path, val_path, test_path = get_cats_dogs_paths()
        print(f"  训练集: {train_path}")
        print(f"  验证集: {val_path}")
        print(f"  测试集: {test_path}")
    except Exception as e:
        print(f"  ❌ 获取路径失败: {e}")