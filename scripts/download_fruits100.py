#!/usr/bin/env python3
"""
在AutoDL上下载fruits100数据集
使用ModelScope SDK进行下载，国内速度更快
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_fruits100_dataset():
    """下载fruits100数据集"""
    try:
        # 安装modelscope
        logger.info("安装ModelScope...")
        os.system("pip3 install modelscope")
        
        # 导入modelscope
        from modelscope.msdatasets import MsDataset
        
        # 设置数据集下载路径
        data_dir = "/root/ai-learning/data"
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("开始下载fruits100数据集...")
        logger.info("数据集将保存到: /root/ai-learning/data/fruits100")
        
        # 下载数据集
        dataset = MsDataset.load(
            'tany0699/fruits100',
            cache_dir=data_dir,
            split='train'  # 可以根据需要调整
        )
        
        logger.info("✅ fruits100数据集下载完成!")
        logger.info(f"数据集路径: {data_dir}/fruits100")
        
        # 检查下载的文件
        fruits_path = os.path.join(data_dir, "fruits100")
        if os.path.exists(fruits_path):
            logger.info("数据集文件结构:")
            for root, dirs, files in os.walk(fruits_path):
                level = root.replace(fruits_path, '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # 只显示前5个文件
                    logger.info(f"{subindent}{file}")
                if len(files) > 5:
                    logger.info(f"{subindent}... 还有 {len(files) - 5} 个文件")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据集下载失败: {e}")
        return False

def download_with_git_lfs():
    """备用方案：使用git lfs下载"""
    try:
        logger.info("使用Git LFS下载数据集...")
        
        # 切换到数据目录
        data_dir = "/root/ai-learning/data"
        os.makedirs(data_dir, exist_ok=True)
        os.chdir(data_dir)
        
        # 使用git lfs下载
        commands = [
            "git lfs install",
            "GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/datasets/tany0699/fruits100.git",
            "cd fruits100",
            "git lfs fetch --include='*.jpg' --include='*.png' --include='*.jpeg'",
            "git lfs checkout",
            "rm -rf .git"  # 清理git目录节省空间
        ]
        
        for cmd in commands:
            logger.info(f"执行: {cmd}")
            result = os.system(cmd)
            if result != 0:
                logger.warning(f"命令执行可能有问题: {cmd}")
        
        logger.info("✅ Git LFS下载完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Git LFS下载失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("开始下载fruits100数据集到AutoDL...")
    
    # 首先尝试ModelScope SDK
    if download_fruits100_dataset():
        logger.info("✅ 使用ModelScope SDK下载成功!")
    else:
        logger.info("ModelScope SDK下载失败，尝试Git LFS方案...")
        if download_with_git_lfs():
            logger.info("✅ 使用Git LFS下载成功!")
        else:
            logger.error("❌ 所有下载方案都失败了")
            sys.exit(1)
    
    logger.info("🎉 数据集下载完成，可以开始训练了!")