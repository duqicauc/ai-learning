#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for data configuration
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_root)

def test_data_config():
    """Test the data configuration"""
    print("🧪 Testing Data Configuration...")
    
    try:
        # 简单的数据配置测试
        data_root = Path(project_root) / "data"
        fruits_root = data_root / "fruits100"
        
        train_dir = fruits_root / "train"
        val_dir = fruits_root / "val"
        test_dir = fruits_root / "test"
        
        print("✅ Data configuration created successfully")
        print(f"📁 Data paths:")
        print(f"   Data root: {data_root}")
        print(f"   Fruits root: {fruits_root}")
        print(f"   Train dir: {train_dir}")
        print(f"   Val dir: {val_dir}")
        print(f"   Test dir: {test_dir}")
        
        # 检查路径是否存在
        print(f"\n📊 Path existence check:")
        print(f"   Data root exists: {data_root.exists()}")
        print(f"   Fruits root exists: {fruits_root.exists()}")
        print(f"   Train dir exists: {train_dir.exists()}")
        print(f"   Val dir exists: {val_dir.exists()}")
        print(f"   Test dir exists: {test_dir.exists()}")
        
        if fruits_root.exists():
            print(f"\n📂 Fruits directory contents:")
            for item in fruits_root.iterdir():
                if item.is_dir():
                    print(f"   📁 {item.name}/")
                else:
                    print(f"   📄 {item.name}")
        
        print("\n🎉 Data configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_config()