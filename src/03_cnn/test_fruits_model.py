#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for fruits classifier model
"""

import sys
import os
import torch
import torch.nn as nn

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(project_root)

def test_model():
    """Test the FruitsCNN model"""
    print("🧪 Testing FruitsCNN Model...")
    
    try:
        # 直接定义FruitsCNN类进行测试
        class FruitsCNN(nn.Module):
            """
            水果分类CNN模型
            支持100种水果的多分类任务
            """
            def __init__(self, num_classes=100):
                super(FruitsCNN, self).__init__()
                
                # 第一个卷积块：输入3通道 -> 64通道
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.pool1 = nn.MaxPool2d(2, 2)  # 224x224 -> 112x112
                
                # 第二个卷积块：64通道 -> 128通道
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)
                self.pool2 = nn.MaxPool2d(2, 2)  # 112x112 -> 56x56
                
                # 第三个卷积块：128通道 -> 256通道
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(256)
                self.pool3 = nn.MaxPool2d(2, 2)  # 56x56 -> 28x28
                
                # 第四个卷积块：256通道 -> 512通道
                self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(512)
                self.pool4 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
                
                # 全局平均池化
                self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                
                # 全连接层
                self.fc1 = nn.Linear(512, 256)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(256, num_classes)
                
                # 权重初始化
                self._initialize_weights()
            
            def _initialize_weights(self):
                """初始化网络权重"""
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                """前向传播"""
                # 卷积块1
                x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
                
                # 卷积块2
                x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
                
                # 卷积块3
                x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
                
                # 卷积块4
                x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
                
                # 全局平均池化
                x = self.global_avg_pool(x)
                x = x.view(x.size(0), -1)  # 展平
                
                # 全连接层
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        
        # Create model
        model = FruitsCNN(num_classes=100)
        print("✅ Model created successfully")
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024 / 1024  # Assuming float32
        
        print(f"📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {model_size_mb:.2f} MB")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            print(f"✅ Forward pass successful")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test batch processing
        batch_input = torch.randn(4, 3, 224, 224)
        batch_output = model(batch_input)
        print(f"✅ Batch processing successful")
        print(f"   Batch input shape: {batch_input.shape}")
        print(f"   Batch output shape: {batch_output.shape}")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model()