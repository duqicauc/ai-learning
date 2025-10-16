#!/usr/bin/env python3
"""
数据集验证和统计脚本

验证数据集的完整性、统计信息，并生成数据质量报告。

使用方法:
    python scripts/validate_data.py --dataset cats_and_dogs
    python scripts/validate_data.py --all
    python scripts/validate_data.py --dataset cats_and_dogs --detailed
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import time
from PIL import Image
import hashlib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def validate_dataset(self, dataset_name: str, detailed: bool = False) -> Dict:
        """
        验证指定数据集
        
        Args:
            dataset_name: 数据集名称
            detailed: 是否进行详细验证
            
        Returns:
            Dict: 验证结果
        """
        dataset_path = self.data_root / dataset_name
        
        if not dataset_path.exists():
            return {
                "dataset": dataset_name,
                "status": "error",
                "message": f"数据集目录不存在: {dataset_path}"
            }
        
        print(f"🔍 验证数据集: {dataset_name}")
        print(f"   路径: {dataset_path}")
        
        result = {
            "dataset": dataset_name,
            "path": str(dataset_path),
            "status": "success",
            "timestamp": time.time(),
            "structure": {},
            "statistics": {},
            "issues": []
        }
        
        try:
            # 基本结构验证
            result["structure"] = self._validate_structure(dataset_path)
            
            # 统计信息
            result["statistics"] = self._collect_statistics(dataset_path)
            
            # 详细验证
            if detailed:
                result["detailed"] = self._detailed_validation(dataset_path)
            
            # 检查问题
            result["issues"] = self._check_issues(dataset_path, result["statistics"])
            
            # 确定整体状态
            if result["issues"]:
                result["status"] = "warning" if any(issue["level"] == "error" for issue in result["issues"]) else "warning"
            
            self._print_validation_report(result)
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            print(f"❌ 验证失败: {e}")
        
        return result
    
    def _validate_structure(self, dataset_path: Path) -> Dict:
        """验证数据集目录结构"""
        structure = {
            "has_train": False,
            "has_val": False,
            "has_test": False,
            "train_classes": [],
            "val_classes": [],
            "test_classes": [],
            "readme_exists": False
        }
        
        # 检查主要目录
        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        test_path = dataset_path / "test"
        readme_path = dataset_path / "README.md"
        
        structure["has_train"] = train_path.exists() and train_path.is_dir()
        structure["has_val"] = val_path.exists() and val_path.is_dir()
        structure["has_test"] = test_path.exists() and test_path.is_dir()
        structure["readme_exists"] = readme_path.exists()
        
        # 检查类别目录
        if structure["has_train"]:
            structure["train_classes"] = [d.name for d in train_path.iterdir() if d.is_dir()]
        
        if structure["has_val"]:
            structure["val_classes"] = [d.name for d in val_path.iterdir() if d.is_dir()]
        
        if structure["has_test"]:
            structure["test_classes"] = [d.name for d in test_path.iterdir() if d.is_dir()]
        
        return structure
    
    def _collect_statistics(self, dataset_path: Path) -> Dict:
        """收集数据集统计信息"""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "file_formats": Counter(),
            "splits": {},
            "class_distribution": {}
        }
        
        for split in ["train", "val", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                split_stats = self._analyze_split(split_path)
                stats["splits"][split] = split_stats
                stats["total_files"] += split_stats["total_files"]
                stats["total_size_mb"] += split_stats["total_size_mb"]
                
                # 合并文件格式统计
                for fmt, count in split_stats["file_formats"].items():
                    stats["file_formats"][fmt] += count
                
                # 合并类别分布
                for cls, count in split_stats["class_distribution"].items():
                    if cls not in stats["class_distribution"]:
                        stats["class_distribution"][cls] = {}
                    stats["class_distribution"][cls][split] = count
        
        return stats
    
    def _analyze_split(self, split_path: Path) -> Dict:
        """分析单个数据分割"""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "file_formats": Counter(),
            "class_distribution": {},
            "avg_file_size_kb": 0,
            "image_dimensions": []
        }
        
        total_size_bytes = 0
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_files = []
            
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    class_files.append(file_path)
                    file_size = file_path.stat().st_size
                    total_size_bytes += file_size
                    
                    # 统计文件格式
                    stats["file_formats"][file_path.suffix.lower()] += 1
            
            stats["class_distribution"][class_name] = len(class_files)
            stats["total_files"] += len(class_files)
        
        stats["total_size_mb"] = total_size_bytes / (1024 * 1024)
        if stats["total_files"] > 0:
            stats["avg_file_size_kb"] = total_size_bytes / stats["total_files"] / 1024
        
        return stats
    
    def _detailed_validation(self, dataset_path: Path) -> Dict:
        """详细验证（检查图片完整性等）"""
        detailed = {
            "corrupted_files": [],
            "duplicate_files": [],
            "unusual_dimensions": [],
            "file_integrity": {"passed": 0, "failed": 0}
        }
        
        print("   🔍 进行详细验证...")
        
        file_hashes = {}
        
        for split in ["train", "val", "test"]:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
            
            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                for file_path in class_dir.iterdir():
                    if file_path.suffix.lower() not in self.supported_formats:
                        continue
                    
                    try:
                        # 检查图片完整性
                        with Image.open(file_path) as img:
                            img.verify()  # 验证图片
                            
                            # 重新打开获取尺寸信息
                            with Image.open(file_path) as img2:
                                width, height = img2.size
                                
                                # 检查异常尺寸
                                if width < 32 or height < 32 or width > 5000 or height > 5000:
                                    detailed["unusual_dimensions"].append({
                                        "file": str(file_path.relative_to(dataset_path)),
                                        "dimensions": f"{width}x{height}"
                                    })
                        
                        # 计算文件哈希检查重复
                        file_hash = self._calculate_file_hash(file_path)
                        if file_hash in file_hashes:
                            detailed["duplicate_files"].append({
                                "file1": str(file_hashes[file_hash].relative_to(dataset_path)),
                                "file2": str(file_path.relative_to(dataset_path))
                            })
                        else:
                            file_hashes[file_hash] = file_path
                        
                        detailed["file_integrity"]["passed"] += 1
                        
                    except Exception as e:
                        detailed["corrupted_files"].append({
                            "file": str(file_path.relative_to(dataset_path)),
                            "error": str(e)
                        })
                        detailed["file_integrity"]["failed"] += 1
        
        return detailed
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _check_issues(self, dataset_path: Path, statistics: Dict) -> List[Dict]:
        """检查数据集问题"""
        issues = []
        
        # 检查是否有训练集
        if "train" not in statistics["splits"]:
            issues.append({
                "level": "error",
                "type": "missing_split",
                "message": "缺少训练集目录"
            })
        
        # 检查是否有验证集
        if "val" not in statistics["splits"]:
            issues.append({
                "level": "warning",
                "type": "missing_split",
                "message": "缺少验证集目录"
            })
        
        # 检查类别一致性
        if "train" in statistics["splits"] and "val" in statistics["splits"]:
            train_classes = set(statistics["class_distribution"].keys())
            val_classes = set()
            for cls, splits in statistics["class_distribution"].items():
                if "val" in splits:
                    val_classes.add(cls)
            
            if train_classes != val_classes:
                issues.append({
                    "level": "warning",
                    "type": "class_mismatch",
                    "message": f"训练集和验证集类别不一致: 训练集{train_classes}, 验证集{val_classes}"
                })
        
        # 检查类别平衡
        for split, split_stats in statistics["splits"].items():
            class_counts = list(split_stats["class_distribution"].values())
            if len(class_counts) > 1:
                min_count = min(class_counts)
                max_count = max(class_counts)
                if max_count / min_count > 3:  # 不平衡阈值
                    issues.append({
                        "level": "warning",
                        "type": "class_imbalance",
                        "message": f"{split}集类别不平衡: 最少{min_count}个样本, 最多{max_count}个样本"
                    })
        
        # 检查文件数量
        if statistics["total_files"] < 100:
            issues.append({
                "level": "warning",
                "type": "small_dataset",
                "message": f"数据集较小，仅有{statistics['total_files']}个文件"
            })
        
        return issues
    
    def _print_validation_report(self, result: Dict) -> None:
        """打印验证报告"""
        print(f"\n📊 验证报告: {result['dataset']}")
        print("=" * 50)
        
        # 状态
        status_emoji = {"success": "✅", "warning": "⚠️", "error": "❌"}
        print(f"状态: {status_emoji.get(result['status'], '❓')} {result['status'].upper()}")
        
        # 结构信息
        structure = result["structure"]
        print(f"\n📁 目录结构:")
        print(f"   训练集: {'✅' if structure['has_train'] else '❌'}")
        print(f"   验证集: {'✅' if structure['has_val'] else '❌'}")
        print(f"   测试集: {'✅' if structure['has_test'] else '❌'}")
        print(f"   README: {'✅' if structure['readme_exists'] else '❌'}")
        
        # 统计信息
        stats = result["statistics"]
        print(f"\n📈 统计信息:")
        print(f"   总文件数: {stats['total_files']:,}")
        print(f"   总大小: {stats['total_size_mb']:.1f} MB")
        print(f"   文件格式: {dict(stats['file_formats'])}")
        
        # 分割统计
        for split, split_stats in stats["splits"].items():
            print(f"\n   {split.upper()}集:")
            print(f"     文件数: {split_stats['total_files']:,}")
            print(f"     大小: {split_stats['total_size_mb']:.1f} MB")
            print(f"     平均文件大小: {split_stats['avg_file_size_kb']:.1f} KB")
            print(f"     类别分布: {split_stats['class_distribution']}")
        
        # 问题报告
        if result["issues"]:
            print(f"\n⚠️ 发现问题:")
            for issue in result["issues"]:
                level_emoji = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
                print(f"   {level_emoji.get(issue['level'], '❓')} {issue['message']}")
        else:
            print(f"\n✅ 未发现问题")
        
        # 详细验证结果
        if "detailed" in result:
            detailed = result["detailed"]
            print(f"\n🔍 详细验证:")
            print(f"   文件完整性: {detailed['file_integrity']['passed']} 通过, {detailed['file_integrity']['failed']} 失败")
            
            if detailed["corrupted_files"]:
                print(f"   损坏文件: {len(detailed['corrupted_files'])} 个")
                for corrupt in detailed["corrupted_files"][:5]:  # 只显示前5个
                    print(f"     - {corrupt['file']}: {corrupt['error']}")
            
            if detailed["duplicate_files"]:
                print(f"   重复文件: {len(detailed['duplicate_files'])} 对")
                for dup in detailed["duplicate_files"][:3]:  # 只显示前3对
                    print(f"     - {dup['file1']} ≈ {dup['file2']}")
            
            if detailed["unusual_dimensions"]:
                print(f"   异常尺寸: {len(detailed['unusual_dimensions'])} 个")
                for unusual in detailed["unusual_dimensions"][:3]:  # 只显示前3个
                    print(f"     - {unusual['file']}: {unusual['dimensions']}")
    
    def validate_all_datasets(self, detailed: bool = False) -> Dict[str, Dict]:
        """验证所有数据集"""
        results = {}
        
        if not self.data_root.exists():
            print(f"❌ 数据根目录不存在: {self.data_root}")
            return results
        
        datasets = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        
        if not datasets:
            print(f"ℹ️ 数据根目录为空: {self.data_root}")
            return results
        
        print(f"🔍 验证所有数据集 (共 {len(datasets)} 个)")
        print("=" * 60)
        
        for dataset in datasets:
            print(f"\n{'='*20} {dataset} {'='*20}")
            results[dataset] = self.validate_dataset(dataset, detailed)
        
        # 总结报告
        print(f"\n📋 总结报告")
        print("=" * 60)
        
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        warning_count = sum(1 for r in results.values() if r["status"] == "warning")
        error_count = sum(1 for r in results.values() if r["status"] == "error")
        
        print(f"✅ 成功: {success_count}")
        print(f"⚠️ 警告: {warning_count}")
        print(f"❌ 错误: {error_count}")
        
        return results
    
    def save_report(self, results: Dict, output_path: str = "data_validation_report.json") -> None:
        """保存验证报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 验证报告已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集验证工具")
    parser.add_argument("--dataset", type=str, help="要验证的数据集名称")
    parser.add_argument("--all", action="store_true", help="验证所有数据集")
    parser.add_argument("--detailed", action="store_true", help="进行详细验证")
    parser.add_argument("--data-root", type=str, default="data", help="数据根目录")
    parser.add_argument("--save-report", type=str, help="保存报告到指定文件")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.data_root)
    
    if args.all:
        results = validator.validate_all_datasets(args.detailed)
    elif args.dataset:
        results = {args.dataset: validator.validate_dataset(args.dataset, args.detailed)}
    else:
        parser.print_help()
        return
    
    if args.save_report:
        validator.save_report(results, args.save_report)

if __name__ == "__main__":
    main()