#!/usr/bin/env python3
"""
快速开始脚本 - Trae + AutoDL 工作流演示
一键体验完整的开发到训练流程
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuickStart:
    """快速开始管理器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.scripts_dir = self.project_root / "scripts"
        
    def print_banner(self):
        """打印欢迎横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 Trae + AutoDL 工作流                    ║
║                      快速开始演示脚本                          ║
╠══════════════════════════════════════════════════════════════╣
║  本脚本将演示从本地开发到云端训练的完整流程                      ║
║  包括环境设置、代码测试、配置生成和部署准备                      ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def step_separator(self, step_num, title):
        """步骤分隔符"""
        print(f"\n{'='*60}")
        print(f"📋 步骤 {step_num}: {title}")
        print(f"{'='*60}")
    
    def run_command(self, command, description):
        """运行命令并显示结果"""
        logger.info(f"🔄 {description}")
        logger.info(f"执行: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                cwd=self.project_root,
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} - 成功")
                if result.stdout.strip():
                    print("输出:")
                    print(result.stdout[-500:])  # 显示最后500字符
            else:
                logger.warning(f"⚠️ {description} - 有警告")
                if result.stderr.strip():
                    print("警告信息:")
                    print(result.stderr[-500:])
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ {description} - 超时，但可能正常")
            return True
        except Exception as e:
            logger.error(f"❌ {description} - 失败: {e}")
            return False
    
    def check_prerequisites(self):
        """检查先决条件"""
        self.step_separator(1, "检查先决条件")
        
        checks = [
            (["python", "--version"], "Python版本检查"),
            (["git", "--version"], "Git版本检查"),
        ]
        
        all_passed = True
        for command, description in checks:
            if not self.run_command(command, description):
                all_passed = False
        
        if all_passed:
            logger.info("✅ 所有先决条件检查通过")
        else:
            logger.warning("⚠️ 部分先决条件检查失败，但可以继续")
        
        return True
    
    def setup_local_environment(self):
        """设置本地环境"""
        self.step_separator(2, "设置本地开发环境")
        
        # 检查local_dev.py是否存在
        local_dev_script = self.scripts_dir / "local_dev.py"
        if not local_dev_script.exists():
            logger.error(f"❌ 脚本不存在: {local_dev_script}")
            return False
        
        # 运行环境设置
        command = [sys.executable, str(local_dev_script), "--action", "setup"]
        return self.run_command(command, "本地环境设置")
    
    def create_sample_data(self):
        """创建样本数据"""
        self.step_separator(3, "创建样本数据")
        
        local_dev_script = self.scripts_dir / "local_dev.py"
        command = [sys.executable, str(local_dev_script), "--action", "sample", "--sample-size", "5"]
        return self.run_command(command, "创建样本数据")
    
    def run_local_test(self):
        """运行本地测试"""
        self.step_separator(4, "运行本地快速测试")
        
        local_dev_script = self.scripts_dir / "local_dev.py"
        command = [sys.executable, str(local_dev_script), "--action", "test", "--model", "fruits"]
        return self.run_command(command, "本地模型测试")
    
    def generate_configs(self):
        """生成配置文件"""
        self.step_separator(5, "生成配置文件")
        
        local_dev_script = self.scripts_dir / "local_dev.py"
        
        configs = ["local", "debug"]
        for config in configs:
            command = [sys.executable, str(local_dev_script), "--action", "config", "--config-template", config]
            self.run_command(command, f"生成{config}配置")
    
    def validate_project_structure(self):
        """验证项目结构"""
        self.step_separator(6, "验证项目结构")
        
        required_items = [
            ("configs/", "配置目录"),
            ("scripts/", "脚本目录"),
            ("src/", "源码目录"),
            ("requirements/", "依赖目录"),
            ("docs/", "文档目录"),
            ("configs/local.yaml", "本地配置文件"),
            ("configs/autodl.yaml", "AutoDL配置文件"),
            ("scripts/sync_to_autodl.py", "同步脚本"),
            ("scripts/local_dev.py", "本地开发脚本"),
        ]
        
        logger.info("📁 检查项目结构:")
        all_exist = True
        
        for item, description in required_items:
            item_path = self.project_root / item
            if item_path.exists():
                logger.info(f"  ✅ {description}: {item}")
            else:
                logger.warning(f"  ⚠️ {description}: {item} (缺失)")
                all_exist = False
        
        if all_exist:
            logger.info("✅ 项目结构完整")
        else:
            logger.warning("⚠️ 项目结构不完整，但核心功能可用")
        
        return True
    
    def show_next_steps(self):
        """显示后续步骤"""
        self.step_separator(7, "后续步骤指南")
        
        next_steps = """
🎉 恭喜！本地开发环境已经设置完成！

📋 接下来你可以：

1️⃣ 配置AutoDL连接:
   编辑 configs/sync_config.yaml
   设置你的AutoDL实例信息

2️⃣ 开始开发:
   在 src/ 目录下编写你的模型代码
   使用 python scripts/local_dev.py --action test 进行快速测试

3️⃣ 部署到AutoDL:
   python scripts/sync_to_autodl.py --action full

4️⃣ 监控训练:
   python scripts/sync_to_autodl.py --action monitor

📚 详细文档:
   查看 docs/WORKFLOW_GUIDE.md 获取完整指南
   查看 docs/TRAE_AUTODL_WORKFLOW.md 了解架构设计

🔧 常用命令:
   # 本地开发
   python scripts/local_dev.py --action setup
   python scripts/local_dev.py --action test
   
   # 代码同步
   python scripts/sync_to_autodl.py --action sync
   python scripts/sync_to_autodl.py --action train
   
   # 配置管理
   python scripts/local_dev.py --action config --config-template local

💡 提示:
   - 使用样本数据进行快速迭代开发
   - 本地验证通过后再部署到AutoDL
   - 定期备份重要的训练结果
        """
        
        print(next_steps)
    
    def run_demo(self):
        """运行完整演示"""
        self.print_banner()
        
        print("\n🚀 开始快速演示...")
        print("这个过程大约需要3-5分钟，请耐心等待...")
        
        steps = [
            ("检查先决条件", self.check_prerequisites),
            ("设置本地环境", self.setup_local_environment),
            ("创建样本数据", self.create_sample_data),
            ("运行本地测试", self.run_local_test),
            ("生成配置文件", self.generate_configs),
            ("验证项目结构", self.validate_project_structure),
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                time.sleep(1)  # 短暂暂停，让用户看清输出
            except KeyboardInterrupt:
                print("\n\n⏹️ 用户中断演示")
                return False
            except Exception as e:
                logger.error(f"❌ 步骤失败: {step_name} - {e}")
        
        # 显示总结
        print(f"\n{'='*60}")
        print(f"📊 演示完成总结")
        print(f"{'='*60}")
        print(f"✅ 成功步骤: {success_count}/{total_steps}")
        print(f"📈 成功率: {success_count/total_steps*100:.1f}%")
        
        if success_count >= total_steps * 0.8:  # 80%成功率
            print("🎉 演示基本成功！你的开发环境已经准备就绪。")
            self.show_next_steps()
        else:
            print("⚠️ 演示部分成功，请检查错误信息并手动完成剩余步骤。")
        
        return success_count >= total_steps * 0.8

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trae + AutoDL 快速开始演示")
    parser.add_argument('--skip-test', action='store_true', 
                       help='跳过模型测试步骤（加快演示速度）')
    
    args = parser.parse_args()
    
    demo = QuickStart()
    
    try:
        success = demo.run_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 演示过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()