#!/usr/bin/env python3
"""
AutoDL代码同步和训练工具 - 优化版
支持纯SCP同步，无需Git依赖
"""

import os
import sys
import yaml
import logging
import argparse
import paramiko
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('autodl_sync.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AutoDLSync:
    """AutoDL同步管理器 - 优化版"""
    
    def __init__(self, config_path: str = "configs/sync_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.ssh_client = None
        self.sftp_client = None
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ 配置文件加载成功: {self.config_path}")
                return config
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        # 从环境变量获取敏感信息
        autodl_password = os.getenv('AUTODL_PASSWORD')
        if not autodl_password:
            logger.warning("⚠️  未设置环境变量 AUTODL_PASSWORD，请设置后再使用")
            autodl_password = 'YOUR_PASSWORD_HERE'  # 占位符
        
        return {
            'autodl': {
                'host': os.getenv('AUTODL_HOST', 'connect.westc.gpuhub.com'),
                'port': int(os.getenv('AUTODL_PORT', '41852')),
                'username': os.getenv('AUTODL_USERNAME', 'root'),
                'password': autodl_password,
                'remote_path': os.getenv('AUTODL_REMOTE_PATH', '/root/ai-learning'),
                'local_path': '.',
                'exclude_patterns': [
                    '.git/',
                    '__pycache__/',
                    '*.pyc',
                    '.vscode/',
                    'node_modules/',
                    'data/',
                    'outputs/',
                    '*.log'
                ]
            },
            'sync': {
                'method': 'scp',  # 改为SCP同步
                'compress': True,
                'preserve_permissions': True,
                'delete_remote': False  # 是否删除远程多余文件
            }
        }
    
    def connect_ssh(self) -> bool:
        """建立SSH连接"""
        try:
            autodl_config = self.config['autodl']
            
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            logger.info(f"🔗 连接到AutoDL: {autodl_config['host']}:{autodl_config['port']}")
            
            self.ssh_client.connect(
                hostname=autodl_config['host'],
                port=autodl_config['port'],
                username=autodl_config['username'],
                password=autodl_config['password'],
                timeout=30
            )
            
            # 创建SFTP客户端
            self.sftp_client = self.ssh_client.open_sftp()
            
            logger.info("✅ SSH连接建立成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ SSH连接失败: {e}")
            return False
    
    def execute_remote_command(self, command: str) -> Tuple[str, str]:
        """执行远程命令"""
        try:
            if not self.ssh_client:
                raise Exception("SSH连接未建立")
            
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            return output, error
            
        except Exception as e:
            logger.error(f"远程命令执行失败: {e}")
            return "", str(e)

    def create_exclude_file(self) -> str:
        """创建排除文件列表"""
        exclude_file = '.sync_exclude'
        exclude_patterns = self.config['autodl']['exclude_patterns']
        
        try:
            with open(exclude_file, 'w', encoding='utf-8') as f:
                for pattern in exclude_patterns:
                    f.write(f"{pattern}\n")
            
            logger.info(f"✅ 排除文件列表创建: {exclude_file}")
            return exclude_file
            
        except Exception as e:
            logger.error(f"创建排除文件失败: {e}")
            return ""

    def sync_code_scp(self) -> bool:
        """使用SCP同步代码"""
        try:
            autodl_config = self.config['autodl']
            sync_config = self.config['sync']
            
            local_path = autodl_config['local_path']
            remote_path = autodl_config['remote_path']
            host = autodl_config['host']
            port = autodl_config['port']
            username = autodl_config['username']
            
            logger.info("🚀 开始SCP代码同步...")
            
            # 创建排除文件
            exclude_file = self.create_exclude_file()
            
            # 构建rsync命令（通过SSH）
            rsync_cmd = [
                'rsync',
                '-avz',  # archive, verbose, compress
                '--progress',
                '--delete' if sync_config.get('delete_remote', False) else '--no-delete',
                f'--exclude-from={exclude_file}' if exclude_file else '',
                '-e', f'ssh -p {port} -o StrictHostKeyChecking=no',
                f'{local_path}/',
                f'{username}@{host}:{remote_path}/'
            ]
            
            # 过滤空参数
            rsync_cmd = [arg for arg in rsync_cmd if arg]
            
            logger.info(f"执行同步命令: {' '.join(rsync_cmd)}")
            
            # 执行同步
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                logger.info("✅ 代码同步成功")
                logger.info(f"同步输出: {result.stdout}")
                
                # 清理排除文件
                if exclude_file and os.path.exists(exclude_file):
                    os.remove(exclude_file)
                
                return True
            else:
                logger.error(f"❌ 代码同步失败: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("❌ rsync命令未找到，请安装rsync")
            return self.sync_code_sftp()  # 降级到SFTP
        except Exception as e:
            logger.error(f"❌ SCP同步异常: {e}")
            return self.sync_code_sftp()  # 降级到SFTP

    def sync_code_sftp(self) -> bool:
        """使用SFTP同步代码（备用方案）"""
        try:
            if not self.sftp_client:
                logger.error("SFTP客户端未初始化")
                return False
            
            autodl_config = self.config['autodl']
            local_path = Path(autodl_config['local_path'])
            remote_path = autodl_config['remote_path']
            exclude_patterns = autodl_config['exclude_patterns']
            
            logger.info("🚀 开始SFTP代码同步...")
            
            # 确保远程目录存在
            self.execute_remote_command(f"mkdir -p {remote_path}")
            
            # 递归上传文件
            uploaded_count = 0
            for local_file in local_path.rglob('*'):
                if local_file.is_file():
                    # 检查是否应该排除
                    relative_path = local_file.relative_to(local_path)
                    should_exclude = any(
                        self._match_pattern(str(relative_path), pattern)
                        for pattern in exclude_patterns
                    )
                    
                    if should_exclude:
                        continue
                    
                    # 构建远程路径
                    remote_file_path = f"{remote_path}/{relative_path}".replace('\\', '/')
                    remote_dir = os.path.dirname(remote_file_path)
                    
                    # 确保远程目录存在
                    try:
                        self.sftp_client.stat(remote_dir)
                    except FileNotFoundError:
                        self.execute_remote_command(f"mkdir -p {remote_dir}")
                    
                    # 上传文件
                    try:
                        self.sftp_client.put(str(local_file), remote_file_path)
                        uploaded_count += 1
                        if uploaded_count % 10 == 0:
                            logger.info(f"已上传 {uploaded_count} 个文件...")
                    except Exception as e:
                        logger.warning(f"上传文件失败 {local_file}: {e}")
            
            logger.info(f"✅ SFTP同步完成，共上传 {uploaded_count} 个文件")
            return True
            
        except Exception as e:
            logger.error(f"❌ SFTP同步异常: {e}")
            return False

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """匹配排除模式"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern) or pattern in path

    def sync_code(self) -> bool:
        """代码同步主入口"""
        sync_method = self.config['sync'].get('method', 'scp')
        
        if sync_method == 'scp':
            return self.sync_code_scp()
        elif sync_method == 'sftp':
            return self.sync_code_sftp()
        else:
            logger.error(f"不支持的同步方法: {sync_method}")
            return False

    def check_environment(self) -> dict:
        """智能检查AutoDL环境状态"""
        try:
            logger.info("🔍 检查AutoDL环境状态...")
            env_status = {}
            
            # 基础系统检查
            basic_checks = [
                ("系统信息", "uname -a"),
                ("磁盘空间", "df -h /"),
                ("内存信息", "free -h"),
                ("GPU信息", "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || echo 'No GPU'")
            ]
            
            for check_name, command in basic_checks:
                output, error = self.execute_remote_command(command)
                if output:
                    logger.info(f"✅ {check_name}: {output.strip()[:100]}...")
                    env_status[check_name] = output.strip()
                else:
                    logger.warning(f"❌ {check_name}: 检查失败")
                    env_status[check_name] = None
            
            # 详细的Python环境检查
            python_checks = [
                ("Python版本", "python --version 2>&1 || python3 --version 2>&1"),
                ("pip版本", "pip --version 2>&1 || pip3 --version 2>&1"),
                ("conda环境", "conda --version 2>&1"),
                ("conda环境列表", "conda env list 2>&1"),
                ("PyTorch", "python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')' 2>&1"),
                ("torchvision", "python -c 'import torchvision; print(f\"torchvision {torchvision.__version__}\")' 2>&1"),
                ("numpy", "python -c 'import numpy; print(f\"numpy {numpy.__version__}\")' 2>&1"),
                ("PIL", "python -c 'from PIL import Image; print(\"PIL available\")' 2>&1"),
                ("matplotlib", "python -c 'import matplotlib; print(f\"matplotlib {matplotlib.__version__}\")' 2>&1")
            ]
            
            for check_name, command in python_checks:
                output, error = self.execute_remote_command(command)
                if output and "error" not in output.lower() and "traceback" not in output.lower():
                    logger.info(f"✅ {check_name}: {output.strip()}")
                    env_status[check_name] = True
                else:
                    logger.warning(f"❌ {check_name}: 未安装或有问题")
                    env_status[check_name] = False
            
            # 检查常用路径
            path_checks = [
                ("miniconda路径", "ls -la /root/miniconda3/bin/conda 2>&1"),
                ("Python路径", "which python 2>&1"),
                ("pip路径", "which pip 2>&1"),
                ("CUDA路径", "ls -la /usr/local/cuda*/bin/nvcc 2>&1 || echo 'CUDA not found'")
            ]
            
            for check_name, command in path_checks:
                output, error = self.execute_remote_command(command)
                if output and "not found" not in output.lower() and "no such file" not in output.lower():
                    logger.info(f"✅ {check_name}: 存在")
                    env_status[f"{check_name}_exists"] = True
                else:
                    logger.warning(f"❌ {check_name}: 不存在")
                    env_status[f"{check_name}_exists"] = False
            
            return env_status
            
        except Exception as e:
            logger.error(f"环境检查失败: {e}")
            return {}

    def install_dependencies(self) -> bool:
        """智能安装依赖 - 避免重复安装已有组件"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            logger.info("🔧 开始智能依赖安装...")
            
            # 检查环境状态
            env_status = self.check_environment()
            
            # 设置PATH环境变量
            path_setup = 'export PATH="/root/miniconda3/bin:$PATH"'
            
            # 1. 检查并安装miniconda
            if not env_status.get("conda环境", False) and not env_status.get("miniconda路径_exists", False):
                logger.info("📦 miniconda未检测到，开始安装...")
                install_conda_cmd = f"""
                {path_setup} && 
                if [ ! -d "/root/miniconda3" ]; then
                    echo "下载miniconda..." && 
                    cd /tmp && 
                    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && 
                    echo "安装miniconda..." && 
                    bash miniconda.sh -b -p /root/miniconda3 && 
                    echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc && 
                    rm miniconda.sh && 
                    echo "miniconda安装完成"
                else
                    echo "miniconda目录已存在"
                fi && 
                {path_setup} && 
                conda --version
                """
                output, error = self.execute_remote_command(install_conda_cmd)
                if "conda" in output:
                    logger.info("✅ miniconda安装/验证成功")
                else:
                    logger.warning(f"⚠️ miniconda安装可能有问题: {error}")
            else:
                logger.info("✅ miniconda已存在，跳过安装")
            
            # 2. 检查并配置Python环境
            if not env_status.get("Python版本", False):
                logger.info("📦 Python环境需要配置...")
                python_cmd = f"""
                {path_setup} && 
                python_version=$(python --version 2>&1 | grep -o 'Python [0-9]\\+\\.[0-9]\\+' || echo "none") && 
                if [[ "$python_version" == *"3.9"* ]] || [[ "$python_version" == *"3.10"* ]] || [[ "$python_version" == *"3.11"* ]]; then
                    echo "Python版本合适: $python_version"
                else
                    echo "安装Python 3.9..." && 
                    conda install python=3.9 -y
                fi && 
                python --version
                """
                output, error = self.execute_remote_command(python_cmd)
                if "Python 3" in output:
                    logger.info("✅ Python环境配置成功")
                else:
                    logger.warning(f"⚠️ Python配置可能有问题: {error}")
            else:
                logger.info("✅ Python环境已就绪，跳过配置")
            
            # 3. 检查并安装PyTorch
            if not env_status.get("PyTorch", False):
                logger.info("📦 PyTorch未检测到，开始安装...")
                
                # 检查CUDA版本
                cuda_check_cmd = "nvidia-smi | grep 'CUDA Version' | awk '{print $9}' || echo '11.8'"
                cuda_output, _ = self.execute_remote_command(cuda_check_cmd)
                cuda_version = cuda_output.strip() if cuda_output else "11.8"
                
                # 根据CUDA版本选择PyTorch安装命令
                if cuda_version.startswith("12"):
                    pytorch_install = "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
                elif cuda_version.startswith("11"):
                    pytorch_install = "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
                else:
                    pytorch_install = "conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
                
                pytorch_cmd = f"""
                {path_setup} && 
                echo "检测到CUDA版本: {cuda_version}" && 
                echo "安装PyTorch..." && 
                {pytorch_install} && 
                echo "验证PyTorch安装..." && 
                python -c 'import torch; print(f"PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}, Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}}")'
                """
                output, error = self.execute_remote_command(pytorch_cmd)
                if "PyTorch" in output:
                    logger.info("✅ PyTorch安装成功")
                    if "CUDA: True" in output:
                        logger.info("🚀 GPU支持已启用")
                    else:
                        logger.warning("⚠️ GPU支持未启用，使用CPU模式")
                else:
                    logger.warning(f"⚠️ PyTorch安装可能有问题: {error}")
            else:
                logger.info("✅ PyTorch已安装，跳过安装")
            
            # 4. 安装其他必要的Python包
            missing_packages = []
            for pkg in ["numpy", "PIL", "matplotlib"]:
                if not env_status.get(pkg, False):
                    missing_packages.append(pkg)
            
            if missing_packages:
                logger.info(f"📦 安装缺失的Python包: {', '.join(missing_packages)}")
                package_map = {
                    "PIL": "pillow",
                    "numpy": "numpy",
                    "matplotlib": "matplotlib"
                }
                
                for pkg in missing_packages:
                    install_name = package_map.get(pkg, pkg)
                    install_cmd = f"""
                    {path_setup} && 
                    pip install {install_name} && 
                    python -c 'import {pkg}; print("{pkg} installed successfully")'
                    """
                    output, error = self.execute_remote_command(install_cmd)
                    if "successfully" in output:
                        logger.info(f"✅ {pkg} 安装成功")
                    else:
                        logger.warning(f"⚠️ {pkg} 安装可能有问题: {error}")
            else:
                logger.info("✅ 基础Python包已安装完整")
            
            # 5. 安装项目特定依赖
            logger.info("📦 检查项目依赖...")
            req_check = f"cd {remote_path} && ls -la requirements/ 2>/dev/null || echo 'no requirements dir'"
            output, error = self.execute_remote_command(req_check)
            
            if "autodl.txt" in output:
                logger.info("📦 安装项目特定依赖...")
                install_cmd = f"""
                cd {remote_path} && 
                {path_setup} && 
                echo "安装项目依赖..." && 
                pip install -r requirements/autodl.txt && 
                echo "项目依赖安装完成"
                """
                output, error = self.execute_remote_command(install_cmd)
                
                if "完成" in output:
                    logger.info("✅ 项目依赖安装完成")
                elif error and "error" in error.lower():
                    logger.warning(f"⚠️ 项目依赖安装警告: {error}")
                else:
                    logger.info("✅ 项目依赖处理完成")
            else:
                logger.info("ℹ️ 未找到项目依赖文件，跳过")
            
            # 6. 最终环境验证
            logger.info("🔍 最终环境验证...")
            final_check_cmd = f"""
            {path_setup} && 
            echo "=== 环境验证报告 ===" && 
            echo "Python: $(python --version)" && 
            echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')" && 
            echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')" && 
            echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')" && 
            echo "Working Directory: $(pwd)" && 
            echo "Disk Space: $(df -h / | tail -1 | awk '{{print $4}}')" && 
            echo "=== 验证完成 ==="
            """
            output, error = self.execute_remote_command(final_check_cmd)
            logger.info(f"📊 环境验证结果:\n{output}")
            
            return True
            
        except Exception as e:
            logger.error(f"依赖安装失败: {e}")
            return False

    def prepare_dataset(self, use_local: bool = False) -> bool:
        """数据集准备 - 支持AutoDL数据集和文件存储"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            if use_local:
                logger.info("📁 使用本地小样本数据集")
                return self._prepare_local_dataset()
            
            # 检查AutoDL预置数据集
            logger.info("🔍 检查AutoDL预置数据集...")
            
            # 方案1：检查fruits100数据集（优先文件存储）
            fruits_paths = [
                "/root/autodl-fs/data/fruits100",
                "/root/autodl-fs/fruits100", 
                "/root/data/fruits100"
            ]
            
            fruits_found = False
            for fruits_path in fruits_paths:
                check_cmd = f"ls -la {fruits_path}"
                output, error = self.execute_remote_command(check_cmd)
                
                if "No such file" not in error:
                    logger.info(f"✅ 发现fruits100数据集: {fruits_path}")
                    # 创建软链接
                    link_cmd = f"""
                    cd {remote_path} && 
                    mkdir -p data && 
                    cd data && 
                    ln -sf {fruits_path} ./fruits100 && 
                    ls -la fruits100/
                    """
                    fruits_found = True
                    break
            
            if fruits_found:
                output, error = self.execute_remote_command(link_cmd)
                if "train" in output:
                    logger.info("✅ fruits100数据集链接创建成功")
                    return True
            
            # 方案2：检查其他预置数据集
            preset_datasets = [
                "/root/autodl-fs/data/CIFAR-10",
                "/root/autodl-fs/data/ImageNet", 
                "/root/autodl-fs/CIFAR-10",
                "/root/autodl-fs/ImageNet",
                "/autodl-pub/CIFAR-10",
                "/autodl-pub/ImageNet",
                "/autodl-tmp/fruits100"
            ]
            
            for dataset_path in preset_datasets:
                check_cmd = f"ls -la {dataset_path}"
                output, error = self.execute_remote_command(check_cmd)
                if "No such file" not in error:
                    dataset_name = os.path.basename(dataset_path)
                    logger.info(f"✅ 发现预置数据集: {dataset_name}")
                    
                    # 创建软链接
                    link_cmd = f"""
                    cd {remote_path} && 
                    mkdir -p data && 
                    cd data && 
                    ln -sf {dataset_path} ./{dataset_name.lower()} && 
                    echo "数据集链接创建成功"
                    """
                    self.execute_remote_command(link_cmd)
                    return True
            
            # 方案3：创建测试数据集
            logger.info("📁 创建小样本测试数据集...")
            return self._create_test_dataset()
            
        except Exception as e:
            logger.error(f"数据集准备失败: {e}")
            return False

    def _prepare_local_dataset(self) -> bool:
        """准备本地小样本数据集"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 创建本地小样本数据集结构
            create_cmd = f"""
            cd {remote_path} && 
            mkdir -p data/fruits100_small/{{train,val,test}}/{{apple,banana,orange}} && 
            echo "本地小样本数据集结构创建完成"
            """
            
            output, error = self.execute_remote_command(create_cmd)
            if "完成" in output:
                logger.info("✅ 本地小样本数据集准备完成")
                return True
            else:
                logger.error(f"本地数据集创建失败: {error}")
                return False
                
        except Exception as e:
            logger.error(f"本地数据集准备失败: {e}")
            return False

    def _create_test_dataset(self) -> bool:
        """创建测试数据集"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            create_cmd = f"""
            cd {remote_path} && 
            mkdir -p data/test_fruits/{{train,val}}/{{apple,banana,orange}} && 
            # 创建一些测试图片占位符
            for category in apple banana orange; do
                for split in train val; do
                    for i in {{1..5}}; do
                        touch data/test_fruits/$split/$category/${{category}}_${{i}}.jpg
                    done
                done
            done && 
            echo "测试数据集创建完成" && 
            find data/test_fruits -type f | wc -l
            """
            
            output, error = self.execute_remote_command(create_cmd)
            if "完成" in output:
                logger.info("✅ 测试数据集创建成功")
                return True
            else:
                logger.error(f"测试数据集创建失败: {error}")
                return False
                
        except Exception as e:
            logger.error(f"测试数据集创建失败: {e}")
            return False

    def run_local_test(self) -> bool:
        """在本地运行小样本测试"""
        try:
            logger.info("🧪 开始本地小样本测试...")
            
            # 检查本地测试脚本
            test_script = "src/03_cnn/local_test.py"
            if not os.path.exists(test_script):
                logger.warning("本地测试脚本不存在，创建简单测试...")
                self._create_local_test_script()
            
            # 运行本地测试
            test_cmd = ["python", test_script, "--epochs", "1", "--batch-size", "4"]
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ 本地测试通过")
                logger.info(f"测试输出: {result.stdout}")
                return True
            else:
                logger.error(f"❌ 本地测试失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"本地测试异常: {e}")
            return False

    def _create_local_test_script(self):
        """创建本地测试脚本"""
        test_script_content = '''#!/usr/bin/env python3
"""
本地小样本测试脚本
"""
import torch
import torch.nn as nn
import argparse

def simple_test():
    """简单的PyTorch测试"""
    print("🧪 开始本地测试...")
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    ).to(device)
    
    # 测试前向传播
    x = torch.randn(4, 10).to(device)
    y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("✅ 本地测试通过")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    simple_test()
'''
        
        os.makedirs("src/03_cnn", exist_ok=True)
        with open("src/03_cnn/local_test.py", 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        
        logger.info("✅ 本地测试脚本创建完成")

    def start_training(self, config_name: str = "autodl") -> bool:
        """启动训练"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 检查训练脚本
            script_check = f"cd {remote_path} && ls -la src/03_cnn/fruits_classifier.py"
            output, error = self.execute_remote_command(script_check)
            if error and "No such file" in error:
                logger.error("❌ 训练脚本不存在，请检查代码同步")
                return False
            
            # 启动训练
            training_cmd = f"""
            cd {remote_path} && 
            export PATH="/root/miniconda3/bin:$PATH" && 
            export CUDA_VISIBLE_DEVICES=0 && 
            python src/03_cnn/fruits_classifier.py
            """
            
            logger.info("🚀 启动训练...")
            output, error = self.execute_remote_command(training_cmd)
            
            if error and "error" in error.lower():
                logger.error(f"❌ 训练启动失败: {error}")
                return False
            else:
                logger.info("✅ 训练已启动")
                return True
                
        except Exception as e:
            logger.error(f"❌ 训练启动异常: {e}")
            return False

    def full_sync_and_train(self, config_name: str = "autodl", use_local: bool = False, run_test: bool = True) -> bool:
        """完整的同步和训练流程"""
        logger.info("🚀 开始完整同步和训练流程...")
        
        try:
            # 1. 本地测试（可选）
            if run_test:
                if not self.run_local_test():
                    logger.error("❌ 本地测试失败，终止流程")
                    return False
            
            # 2. 建立SSH连接
            if not self.connect_ssh():
                return False
            
            # 3. 同步代码
            if not self.sync_code():
                return False
            
            # 4. 智能安装依赖
            if not self.install_dependencies():
                return False
            
            # 5. 准备数据集
            if not self.prepare_dataset(use_local):
                return False
            
            # 6. 启动训练
            if not self.start_training(config_name):
                return False
            
            logger.info("✅ 同步和训练启动完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 流程执行失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
            logger.info("SSH连接已关闭")

def main():
    parser = argparse.ArgumentParser(description="AutoDL代码同步和训练工具 - 优化版")
    parser.add_argument('--config', default='configs/sync_config.yaml', 
                       help='同步配置文件路径')
    parser.add_argument('--action', choices=['sync', 'train', 'test', 'full'],
                       default='full', help='执行的操作')
    parser.add_argument('--use-local', action='store_true', 
                       help='使用本地小样本数据集')
    parser.add_argument('--skip-test', action='store_true', 
                       help='跳过本地测试')
    parser.add_argument('--training-config', default='autodl',
                       help='训练配置名称')
    
    args = parser.parse_args()
    
    # 创建同步器
    syncer = AutoDLSync(args.config)
    
    try:
        if args.action == 'sync':
            syncer.connect_ssh()
            syncer.sync_code()
        elif args.action == 'train':
            syncer.connect_ssh()
            syncer.start_training(args.training_config)
        elif args.action == 'test':
            syncer.run_local_test()
        elif args.action == 'full':
            syncer.full_sync_and_train(
                args.training_config, 
                args.use_local, 
                not args.skip_test
            )
    
    finally:
        syncer.close()

if __name__ == "__main__":
    main()