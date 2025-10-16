"""
本地小样本测试脚本

在AutoDL训练前，先在本地进行小样本测试，验证：
- 代码逻辑正确性
- 模型架构合理性
- 数据加载流程
- 训练配置有效性
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_config import data_config, create_small_datasets, validate_all_datasets

class LocalTester:
    """本地小样本测试器"""
    
    def __init__(self, dataset_name: str = "fruits100", epochs: int = 2):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.test_results = {}
        
        print("🧪 本地小样本测试器初始化")
        print(f"📊 数据集: {dataset_name}")
        print(f"🔄 测试轮数: {epochs}")
    
    def prepare_test_environment(self) -> bool:
        """准备测试环境"""
        try:
            print("\n📁 准备测试环境...")
            
            # 确保在本地环境
            if data_config.current_env.value != "local":
                print("⚠️ 当前不在本地环境，强制切换到本地环境")
                data_config.switch_environment("local")
            
            # 创建小样本数据集
            print("📁 创建小样本数据集...")
            create_small_datasets()
            
            # 验证数据集
            print("🔍 验证小样本数据集...")
            if not data_config.validate_dataset(self.dataset_name, use_small=True):
                print("❌ 小样本数据集验证失败")
                return False
            
            print("✅ 测试环境准备完成")
            return True
            
        except Exception as e:
            print(f"❌ 测试环境准备失败: {e}")
            return False
    
    def test_data_loading(self) -> bool:
        """测试数据加载"""
        try:
            print("\n📊 测试数据加载...")
            
            if self.dataset_name == "cats_and_dogs":
                from src.utils.data_config import get_cats_dogs_paths
                train_path, val_path, test_path = get_cats_dogs_paths(use_small=True)
                
                # 检查路径
                if not all([train_path.exists(), val_path.exists()]):
                    print("❌ 数据路径不存在")
                    return False
                
                # 统计文件数量
                train_files = sum(len(list(class_dir.glob("*.jpg"))) 
                                for class_dir in train_path.iterdir() if class_dir.is_dir())
                val_files = sum(len(list(class_dir.glob("*.jpg"))) 
                              for class_dir in val_path.iterdir() if class_dir.is_dir())
                
                print(f"  训练文件: {train_files}")
                print(f"  验证文件: {val_files}")
                
                self.test_results["data_loading"] = {
                    "train_files": train_files,
                    "val_files": val_files,
                    "status": "success"
                }
            
            elif self.dataset_name == "fruits100":
                from src.utils.data_config import get_fruits100_paths
                train_path, val_path, test_path = get_fruits100_paths(use_small=True)
                
                # 检查路径
                if not all([train_path.exists(), val_path.exists()]):
                    print("❌ 数据路径不存在")
                    return False
                
                # 统计类别和文件数量
                train_classes = [d for d in train_path.iterdir() if d.is_dir()]
                val_classes = [d for d in val_path.iterdir() if d.is_dir()]
                
                train_files = sum(len(list(class_dir.glob("*.jpg"))) 
                                for class_dir in train_classes)
                val_files = sum(len(list(class_dir.glob("*.jpg"))) 
                              for class_dir in val_classes)
                
                print(f"  训练类别: {len(train_classes)}")
                print(f"  验证类别: {len(val_classes)}")
                print(f"  训练文件: {train_files}")
                print(f"  验证文件: {val_files}")
                
                self.test_results["data_loading"] = {
                    "train_classes": len(train_classes),
                    "val_classes": len(val_classes),
                    "train_files": train_files,
                    "val_files": val_files,
                    "status": "success"
                }
            
            print("✅ 数据加载测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 数据加载测试失败: {e}")
            self.test_results["data_loading"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_model_creation(self) -> bool:
        """测试模型创建"""
        try:
            print("\n🏗️ 测试模型创建...")
            
            if self.dataset_name == "cats_and_dogs":
                # 导入猫狗分类器
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "cats_dogs_classifier", 
                    project_root / "src" / "03_cnn" / "cats_dogs_classifier.py"
                )
                cats_dogs_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cats_dogs_module)
                CatsDogsClassifier = cats_dogs_module.CatsDogsClassifier
                
                # 创建模型
                model = CatsDogsClassifier()
                
                # 检查模型参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"  总参数: {total_params:,}")
                print(f"  可训练参数: {trainable_params:,}")
                
                self.test_results["model_creation"] = {
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "status": "success"
                }
            
            elif self.dataset_name == "fruits100":
                # 导入水果分类器
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "fruits_classifier", 
                    project_root / "src" / "03_cnn" / "fruits_classifier.py"
                )
                fruits_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fruits_module)
                FruitsClassifier = fruits_module.FruitsClassifier
                
                # 创建模型
                model = FruitsClassifier()
                
                # 检查模型参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"  总参数: {total_params:,}")
                print(f"  可训练参数: {trainable_params:,}")
                
                self.test_results["model_creation"] = {
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "status": "success"
                }
            
            print("✅ 模型创建测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 模型创建测试失败: {e}")
            self.test_results["model_creation"] = {"status": "failed", "error": str(e)}
            return False
    
    def test_training_loop(self) -> bool:
        """测试训练循环"""
        try:
            print(f"\n🔄 测试训练循环 ({self.epochs} 轮)...")
            
            # 创建临时训练脚本
            test_script = self._create_test_script()
            
            if not test_script.exists():
                print("❌ 测试脚本创建失败")
                return False
            
            # 运行测试脚本
            import subprocess
            
            start_time = time.time()
            result = subprocess.run([
                sys.executable, str(test_script)
            ], capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 训练循环测试通过 (耗时: {duration:.1f}秒)")
                
                # 解析输出中的损失值
                output_lines = result.stdout.split('\n')
                losses = []
                for line in output_lines:
                    if "Loss:" in line:
                        try:
                            loss_str = line.split("Loss:")[1].strip().split()[0]
                            losses.append(float(loss_str))
                        except:
                            pass
                
                self.test_results["training_loop"] = {
                    "duration": duration,
                    "losses": losses,
                    "final_loss": losses[-1] if losses else None,
                    "status": "success"
                }
                
                return True
            else:
                print(f"❌ 训练循环测试失败")
                print(f"错误输出: {result.stderr}")
                
                self.test_results["training_loop"] = {
                    "status": "failed", 
                    "error": result.stderr
                }
                return False
            
        except subprocess.TimeoutExpired:
            print("❌ 训练循环测试超时")
            self.test_results["training_loop"] = {"status": "failed", "error": "timeout"}
            return False
        except Exception as e:
            print(f"❌ 训练循环测试失败: {e}")
            self.test_results["training_loop"] = {"status": "failed", "error": str(e)}
            return False
    
    def _create_test_script(self) -> Path:
        """创建临时测试脚本"""
        test_script_path = project_root / "temp_test_script.py"
        
        if self.dataset_name == "cats_and_dogs":
            script_content = f'''
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 动态导入cats_dogs_classifier
spec = importlib.util.spec_from_file_location(
    "cats_dogs_classifier", 
    Path(__file__).parent / "src" / "03_cnn" / "cats_dogs_classifier.py"
)
cats_dogs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cats_dogs_module)
CatsDogsClassifier = cats_dogs_module.CatsDogsClassifier

from src.utils.data_config import get_cats_dogs_paths

# 使用小样本数据集
train_path, val_path, _ = get_cats_dogs_paths(use_small=True)

# 创建分类器
classifier = CatsDogsClassifier()

# 运行训练（小样本，少轮数）
classifier.train(
    train_dir=str(train_path),
    val_dir=str(val_path),
    epochs={self.epochs},
    batch_size=4,  # 小批次
    save_model=False  # 不保存模型
)
'''
        
        elif self.dataset_name == "fruits100":
            script_content = f'''
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 动态导入fruits_classifier
spec = importlib.util.spec_from_file_location(
    "fruits_classifier", 
    Path(__file__).parent / "src" / "03_cnn" / "fruits_classifier.py"
)
fruits_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fruits_module)
FruitsClassifier = fruits_module.FruitsClassifier

from src.utils.data_config import get_fruits100_paths

# 使用小样本数据集
train_path, val_path, _ = get_fruits100_paths(use_small=True)

# 创建分类器
classifier = FruitsClassifier()

# 运行训练（小样本，少轮数）
classifier.train(
    train_dir=str(train_path),
    val_dir=str(val_path),
    epochs={self.epochs},
    batch_size=4,  # 小批次
    save_model=False  # 不保存模型
)
'''
        
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return test_script_path
    
    def run_full_test(self) -> bool:
        """运行完整测试"""
        print("🧪 开始本地小样本测试")
        print("=" * 50)
        
        start_time = time.time()
        
        # 测试步骤
        test_steps = [
            ("环境准备", self.prepare_test_environment),
            ("数据加载", self.test_data_loading),
            ("模型创建", self.test_model_creation),
            ("训练循环", self.test_training_loop)
        ]
        
        success_count = 0
        for step_name, test_func in test_steps:
            print(f"\n🔍 {step_name}测试...")
            if test_func():
                success_count += 1
            else:
                print(f"❌ {step_name}测试失败，停止后续测试")
                break
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 清理临时文件
        temp_script = project_root / "temp_test_script.py"
        if temp_script.exists():
            temp_script.unlink()
        
        # 输出测试结果
        print("\n" + "=" * 50)
        print("🧪 本地小样本测试结果")
        print("=" * 50)
        
        print(f"📊 测试数据集: {self.dataset_name}")
        print(f"⏱️ 总耗时: {total_duration:.1f}秒")
        print(f"✅ 成功步骤: {success_count}/{len(test_steps)}")
        
        # 详细结果
        for step_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            if status == "success":
                print(f"  ✅ {step_name}: 通过")
                if step_name == "training_loop" and "final_loss" in result:
                    print(f"     最终损失: {result['final_loss']:.4f}")
            else:
                print(f"  ❌ {step_name}: 失败")
                if "error" in result:
                    print(f"     错误: {result['error']}")
        
        # 判断是否可以进行AutoDL训练
        all_passed = success_count == len(test_steps)
        if all_passed:
            print("\n🎉 所有测试通过！可以进行AutoDL训练")
        else:
            print("\n⚠️ 部分测试失败，建议修复问题后再进行AutoDL训练")
        
        return all_passed
    
    def generate_report(self) -> str:
        """生成测试报告"""
        report = f"""
# 本地小样本测试报告

## 测试配置
- 数据集: {self.dataset_name}
- 测试轮数: {self.epochs}
- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 测试结果
"""
        
        for step_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            report += f"\n### {step_name}\n"
            report += f"状态: {'✅ 通过' if status == 'success' else '❌ 失败'}\n"
            
            if status == "success":
                if step_name == "data_loading":
                    if "train_files" in result:
                        report += f"- 训练文件: {result['train_files']}\n"
                    if "val_files" in result:
                        report += f"- 验证文件: {result['val_files']}\n"
                    if "train_classes" in result:
                        report += f"- 训练类别: {result['train_classes']}\n"
                        report += f"- 验证类别: {result['val_classes']}\n"
                
                elif step_name == "model_creation":
                    report += f"- 总参数: {result['total_params']:,}\n"
                    report += f"- 可训练参数: {result['trainable_params']:,}\n"
                
                elif step_name == "training_loop":
                    report += f"- 训练耗时: {result['duration']:.1f}秒\n"
                    if result.get("final_loss"):
                        report += f"- 最终损失: {result['final_loss']:.4f}\n"
            
            else:
                if "error" in result:
                    report += f"- 错误信息: {result['error']}\n"
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地小样本测试")
    parser.add_argument("--dataset", default="fruits100", 
                       choices=["cats_and_dogs", "fruits100"],
                       help="测试数据集")
    parser.add_argument("--epochs", type=int, default=2,
                       help="测试训练轮数")
    parser.add_argument("--report", action="store_true",
                       help="生成测试报告")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = LocalTester(dataset_name=args.dataset, epochs=args.epochs)
    
    # 运行测试
    success = tester.run_full_test()
    
    # 生成报告
    if args.report:
        report = tester.generate_report()
        report_path = project_root / f"local_test_report_{args.dataset}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 测试报告已保存: {report_path}")
    
    # 返回退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()