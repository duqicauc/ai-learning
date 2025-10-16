"""
数据预处理和增强工具模块

提供常用的数据预处理、数据增强和数据变换功能。

主要功能:
- 图像预处理管道
- 数据增强策略
- 批量数据处理
- 数据格式转换
- 数据质量检查

作者: AI Learning Project
版本: 1.0.0
"""

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Union
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from collections import defaultdict

class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
        # 预定义的标准化参数
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # 基础变换
        self.basic_transforms = {
            'resize': transforms.Resize(image_size),
            'center_crop': transforms.CenterCrop(image_size),
            'to_tensor': transforms.ToTensor(),
            'normalize': transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        }
    
    def get_train_transforms(self, augmentation_level: str = "medium") -> transforms.Compose:
        """
        获取训练时的数据变换
        
        Args:
            augmentation_level: 增强级别 ("light", "medium", "heavy")
            
        Returns:
            transforms.Compose: 变换组合
        """
        base_transforms = [
            transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if augmentation_level == "light":
            augment_transforms = [
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        elif augmentation_level == "medium":
            augment_transforms = [
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        elif augmentation_level == "heavy":
            augment_transforms = [
                transforms.RandomRotation(degrees=30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            ]
        else:
            augment_transforms = []
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ]
        
        return transforms.Compose(base_transforms + augment_transforms + final_transforms)
    
    def get_val_transforms(self) -> transforms.Compose:
        """获取验证/测试时的数据变换"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
    
    def get_albumentations_transforms(self, is_train: bool = True) -> A.Compose:
        """
        获取Albumentations变换（更强大的数据增强库）
        
        Args:
            is_train: 是否为训练模式
            
        Returns:
            A.Compose: Albumentations变换组合
        """
        if is_train:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
                ToTensorV2(),
            ])

class CustomDataset(Dataset):
    """自定义数据集类，支持Albumentations变换"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label

class DataAugmentationVisualizer:
    """数据增强可视化工具"""
    
    def __init__(self, dataset: Dataset, class_names: List[str]):
        self.dataset = dataset
        self.class_names = class_names
    
    def visualize_augmentations(self, num_samples: int = 4, 
                              num_augmentations: int = 5) -> None:
        """
        可视化数据增强效果
        
        Args:
            num_samples: 样本数量
            num_augmentations: 每个样本的增强版本数量
        """
        fig, axes = plt.subplots(num_samples, num_augmentations + 1, 
                                figsize=(15, 3 * num_samples))
        
        for i in range(num_samples):
            # 获取原始图像
            original_image, label = self.dataset[i]
            
            # 反标准化以便显示
            original_image = self._denormalize(original_image)
            
            # 显示原始图像
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original\n{self.class_names[label]}")
            axes[i, 0].axis('off')
            
            # 显示增强版本
            for j in range(1, num_augmentations + 1):
                aug_image, _ = self.dataset[i]  # 重新获取（会应用随机增强）
                aug_image = self._denormalize(aug_image)
                
                axes[i, j].imshow(aug_image)
                axes[i, j].set_title(f"Augmented {j}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """反标准化张量以便显示"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor.permute(1, 2, 0).numpy()

class DatasetAnalyzer:
    """数据集分析工具"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
    
    def analyze_dataset(self) -> Dict:
        """分析数据集的统计信息"""
        analysis = {
            "class_distribution": defaultdict(int),
            "image_sizes": [],
            "file_formats": defaultdict(int),
            "total_files": 0,
            "total_size_mb": 0
        }
        
        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
            
            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        analysis["class_distribution"][class_name] += 1
                        analysis["file_formats"][img_file.suffix.lower()] += 1
                        analysis["total_files"] += 1
                        analysis["total_size_mb"] += img_file.stat().st_size / (1024 * 1024)
                        
                        # 获取图像尺寸
                        try:
                            with Image.open(img_file) as img:
                                analysis["image_sizes"].append(img.size)
                        except:
                            pass
        
        return dict(analysis)
    
    def plot_class_distribution(self, analysis: Dict) -> None:
        """绘制类别分布图"""
        classes = list(analysis["class_distribution"].keys())
        counts = list(analysis["class_distribution"].values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts)
        plt.title("Class Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_image_size_distribution(self, analysis: Dict) -> None:
        """绘制图像尺寸分布图"""
        if not analysis["image_sizes"]:
            print("No image size data available")
            return
        
        widths = [size[0] for size in analysis["image_sizes"]]
        heights = [size[1] for size in analysis["image_sizes"]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(widths, bins=30, alpha=0.7)
        ax1.set_title("Image Width Distribution")
        ax1.set_xlabel("Width (pixels)")
        ax1.set_ylabel("Frequency")
        
        ax2.hist(heights, bins=30, alpha=0.7)
        ax2.set_title("Image Height Distribution")
        ax2.set_xlabel("Height (pixels)")
        ax2.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()

class BatchProcessor:
    """批量数据处理工具"""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def resize_images(self, target_size: Tuple[int, int], 
                     quality: int = 95) -> None:
        """
        批量调整图像尺寸
        
        Args:
            target_size: 目标尺寸 (width, height)
            quality: JPEG质量 (1-100)
        """
        print(f"Resizing images to {target_size}...")
        
        for img_file in self.input_dir.rglob("*.jpg"):
            try:
                with Image.open(img_file) as img:
                    # 保持宽高比的调整
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # 创建输出路径
                    relative_path = img_file.relative_to(self.input_dir)
                    output_path = self.output_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 保存调整后的图像
                    img.save(output_path, "JPEG", quality=quality)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print("Batch resizing completed!")
    
    def convert_format(self, target_format: str = "JPEG") -> None:
        """
        批量转换图像格式
        
        Args:
            target_format: 目标格式 ("JPEG", "PNG", etc.)
        """
        print(f"Converting images to {target_format}...")
        
        for img_file in self.input_dir.rglob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    with Image.open(img_file) as img:
                        # 转换为RGB模式（JPEG需要）
                        if target_format == "JPEG" and img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # 创建输出路径
                        relative_path = img_file.relative_to(self.input_dir)
                        output_path = self.output_dir / relative_path.with_suffix(f".{target_format.lower()}")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 保存转换后的图像
                        img.save(output_path, target_format)
                        
                except Exception as e:
                    print(f"Error converting {img_file}: {e}")
        
        print("Batch conversion completed!")

def create_data_loaders(train_dataset: Dataset, val_dataset: Dataset,
                       batch_size: int = 32, num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否使用固定内存
        
    Returns:
        Tuple[DataLoader, DataLoader]: 训练和验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def calculate_dataset_stats(dataset: Dataset) -> Tuple[List[float], List[float]]:
    """
    计算数据集的均值和标准差
    
    Args:
        dataset: 数据集
        
    Returns:
        Tuple[List[float], List[float]]: 均值和标准差
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()

# 示例使用
if __name__ == "__main__":
    # 创建预处理器
    preprocessor = ImagePreprocessor(image_size=(224, 224))
    
    # 获取训练变换
    train_transform = preprocessor.get_train_transforms(augmentation_level="medium")
    val_transform = preprocessor.get_val_transforms()
    
    print("数据预处理工具模块已加载")
    print("可用功能:")
    print("- ImagePreprocessor: 图像预处理")
    print("- DataAugmentationVisualizer: 数据增强可视化")
    print("- DatasetAnalyzer: 数据集分析")
    print("- BatchProcessor: 批量处理")
    print("- create_data_loaders: 创建数据加载器")
    print("- calculate_dataset_stats: 计算数据集统计信息")