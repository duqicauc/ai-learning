import torch
import numpy as np


#创建张量数据示例
x=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
b=x.reshape(3,2)
c=b.flatten()
print(c)


np_array=np.array([4.0,5.0,6.0])
y=torch.from_numpy(np_array)
y_np=y.numpy()

#创建制定形状的张量
zeros=torch.rand((3,2))
z=torch.tensor([1,2,3],dtype=torch.float32)


w=torch.tensor(2.0,requires_grad=True)  #定义权重
b=torch.tensor(1.0,requires_grad=True)  #定义偏置

x=torch.tensor(3.0)  #定义输入
y=w*x+b  #向前计算

print('预测值：y=',y)


#设置一个真实的标签数据 y_true=8.0
x=torch.tensor(3.0)  #定义输入
w=torch.tensor(2.0,requires_grad=True)  #定义权重
b=torch.tensor(1.0,requires_grad=True)  #定义偏置
y=w*x+b  #向前计算
y_true=torch.tensor(8.0)
#定义损失函数计算均方误差
loss=(y-y_true)**2
print('损失值：',loss)
loss.backward()
print('权重的梯度：',w.grad)
print('偏置的梯度：',b.grad)