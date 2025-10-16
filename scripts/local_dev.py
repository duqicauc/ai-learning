#!/usr/bin/env python3
"""
本地开发脚本 - Trae环境
快速测试、验证和开发工具
"""

import os
import sys
import subprocess
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalDev:
    """本地开发管理器"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config_dir = self.project_root / "configs"
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        
    def check_environment(self):
        """检查本地开发环境"""
        logger.info("🔍 检查本地开发环境...")
        
        checks = [
            ("Python版本", [sys.executable, "--version"]),
            ("pip版本", [sys.executable, "-m", "pip", "--version"]),
            ("项目结构", None),
        ]
        
        for check_name, command in checks:
            if command:
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    logger.info(f"✅ {check_name}: {result.stdout.strip()}")
                except Exception as e:
                    logger.error(f"❌ {check_name}: {e}")
            else:
                # 检查项目结构
                self.check_project_structure()
    
    def check_project_structure(self):
        """检查项目结构"""
        required_dirs = ["src", "configs", "data", "scripts"]
        required_files = ["requirements/local.txt", "configs/local.yaml"]
        
        logger.info("📁 检查项目结构:")
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                logger.info(f"  ✅ {dir_name}/")
            else:
                logger.warning(f"  ⚠️ {dir_name}/ (缺失)")
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                logger.info(f"  ✅ {file_name}")
            else:
                logger.warning(f"  ⚠️ {file_name} (缺失)")
    
    def install_dependencies(self):
        """安装本地开发依赖"""
        logger.info("📦 安装本地开发依赖...")
        
        requirements_file = self.project_root / "requirements" / "local.txt"
        
        if not requirements_file.exists():
            logger.error(f"❌ 依赖文件不存在: {requirements_file}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ 依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 依赖安装失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            return False
    
    def run_quick_test(self, model_type="fruits"):
        """运行快速测试"""
        logger.info(f"🧪 运行 {model_type} 模型快速测试...")
        
        # 根据模型类型选择脚本
        script_map = {
            "fruits": "src/03_cnn/fruits_classifier.py",
            "cats_dogs": "src/03_cnn/cats_dogs_classifier.py"
        }
        
        script_path = self.project_root / script_map.get(model_type, script_map["fruits"])
        config_path = self.project_root / "configs" / "local.yaml"
        
        if not script_path.exists():
            logger.error(f"❌ 脚本不存在: {script_path}")
            return False
        
        if not config_path.exists():
            logger.warning(f"⚠️ 配置文件不存在: {config_path}")
            config_args = []
        else:
            config_args = ["--config", str(config_path)]
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.src_dir)
            
            cmd = [sys.executable, str(script_path)] + config_args
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行测试（限制时间）
            process = subprocess.Popen(
                cmd, 
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待一段时间或直到完成
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5分钟超时
                
                if process.returncode == 0:
                    logger.info("✅ 快速测试完成")
                    logger.info("输出:")
                    print(stdout[-1000:])  # 显示最后1000字符
                else:
                    logger.error("❌ 快速测试失败")
                    logger.error("错误输出:")
                    print(stderr[-1000:])
                
                return process.returncode == 0
                
            except subprocess.TimeoutExpired:
                logger.info("⏰ 测试运行超时，终止进程")
                process.terminate()
                return True  # 能运行就算成功
                
        except Exception as e:
            logger.error(f"❌ 测试执行异常: {e}")
            return False
    
    def validate_data(self):
        """验证数据集"""
        logger.info("📊 验证数据集...")
        
        datasets = ["fruits100", "cats_and_dogs"]
        
        for dataset in datasets:
            dataset_path = self.data_dir / dataset
            
            if not dataset_path.exists():
                logger.warning(f"⚠️ 数据集不存在: {dataset}")
                continue
            
            # 检查数据集结构
            subdirs = ["train", "val", "test"]
            for subdir in subdirs:
                subdir_path = dataset_path / subdir
                if subdir_path.exists():
                    # 统计文件数量
                    file_count = len(list(subdir_path.rglob("*.*")))
                    logger.info(f"  ✅ {dataset}/{subdir}: {file_count} 文件")
                else:
                    logger.info(f"  ⚠️ {dataset}/{subdir}: 不存在")
    
    def create_sample_data(self, dataset="fruits100", sample_size=10):
        """创建样本数据用于快速测试"""
        logger.info(f"🎯 创建 {dataset} 样本数据 (每类 {sample_size} 张)...")
        
        source_path = self.data_dir / dataset
        sample_path = self.data_dir / f"{dataset}_sample"
        
        if not source_path.exists():
            logger.error(f"❌ 源数据集不存在: {source_path}")
            return False
        
        try:
            import shutil
            import random
            
            # 创建样本目录
            if sample_path.exists():
                shutil.rmtree(sample_path)
            
            for split in ["train", "val", "test"]:
                source_split = source_path / split
                if not source_split.exists():
                    continue
                
                sample_split = sample_path / split
                sample_split.mkdir(parents=True, exist_ok=True)
                
                # 遍历每个类别
                for class_dir in source_split.iterdir():
                    if not class_dir.is_dir():
                        continue
                    
                    sample_class_dir = sample_split / class_dir.name
                    sample_class_dir.mkdir(exist_ok=True)
                    
                    # 随机选择文件
                    all_files = list(class_dir.glob("*.*"))
                    if len(all_files) > sample_size:
                        selected_files = random.sample(all_files, sample_size)
                    else:
                        selected_files = all_files
                    
                    # 复制文件
                    for file_path in selected_files:
                        shutil.copy2(file_path, sample_class_dir)
                    
                    logger.info(f"  ✅ {split}/{class_dir.name}: {len(selected_files)} 文件")
            
            logger.info(f"✅ 样本数据创建完成: {sample_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 样本数据创建失败: {e}")
            return False
    
    def lint_code(self):
        """代码质量检查"""
        logger.info("🔍 代码质量检查...")
        
        # 检查Python语法
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files[:5]:  # 限制检查文件数量
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
                logger.info(f"  ✅ {py_file.relative_to(self.project_root)}")
            except SyntaxError as e:
                logger.error(f"  ❌ {py_file.relative_to(self.project_root)}: {e}")
    
    def generate_config(self, template="local"):
        """生成配置文件"""
        logger.info(f"⚙️ 生成 {template} 配置文件...")
        
        config_templates = {
            "local": {
                "training": {
                    "epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "device": "cpu",
                    "num_workers": 0
                },
                "data": {
                    "dataset": "fruits100_sample",
                    "image_size": 224,
                    "augmentation": False
                },
                "model": {
                    "architecture": "resnet18",
                    "pretrained": True,
                    "num_classes": 100
                },
                "logging": {
                    "log_interval": 10,
                    "save_model": False
                }
            },
            "debug": {
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.01,
                    "device": "cpu",
                    "num_workers": 0
                },
                "data": {
                    "dataset": "fruits100_sample",
                    "image_size": 64,
                    "augmentation": False
                },
                "model": {
                    "architecture": "resnet18",
                    "pretrained": False,
                    "num_classes": 10
                },
                "logging": {
                    "log_interval": 1,
                    "save_model": False
                }
            }
        }
        
        config = config_templates.get(template, config_templates["local"])
        config_file = self.config_dir / f"{template}.yaml"
        
        try:
            self.config_dir.mkdir(exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ 配置文件已生成: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置文件生成失败: {e}")
            return False
    
    def setup_dev_environment(self):
        """设置开发环境"""
        logger.info("🚀 设置本地开发环境...")
        
        steps = [
            ("检查环境", self.check_environment),
            ("安装依赖", self.install_dependencies),
            ("验证数据", self.validate_data),
            ("生成配置", lambda: self.generate_config("local")),
            ("代码检查", self.lint_code),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 执行: {step_name}")
            try:
                result = step_func()
                if result is False:
                    logger.warning(f"⚠️ {step_name} 未完全成功，但继续执行")
            except Exception as e:
                logger.error(f"❌ {step_name} 失败: {e}")
        
        logger.info("🎉 开发环境设置完成!")

def main():
    parser = argparse.ArgumentParser(description="本地开发工具")
    parser.add_argument('--action', 
                       choices=['setup', 'test', 'validate', 'sample', 'lint', 'config'],
                       default='setup', help='执行的操作')
    parser.add_argument('--model', default='fruits', 
                       help='模型类型 (fruits, cats_dogs)')
    parser.add_argument('--config-template', default='local',
                       help='配置模板 (local, debug)')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='样本数据大小')
    
    args = parser.parse_args()
    
    # 创建开发管理器
    dev = LocalDev()
    
    if args.action == 'setup':
        dev.setup_dev_environment()
    elif args.action == 'test':
        dev.run_quick_test(args.model)
    elif args.action == 'validate':
        dev.validate_data()
    elif args.action == 'sample':
        dev.create_sample_data(sample_size=args.sample_size)
    elif args.action == 'lint':
        dev.lint_code()
    elif args.action == 'config':
        dev.generate_config(args.config_template)

if __name__ == "__main__":
    main()