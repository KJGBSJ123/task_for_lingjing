# task_for_lingjing
社团考核用

**2026/11/10**
- #学习残差块的构建*
- ##1.ResNet：  
- 最新的卷积神经网络，用于解决训练轮数增加导致准确度不升反降的问题。  
- 方法：只算残差，然后直接加输入值  
- ##2.残差块的组成：  
- 输入->卷积层1->规范化层1->Relu激活层->卷积层2->规范化层2->(+输入)->Relu激活层  
- 卷积层：用卷积核扫描，并获取信息  
- 激活层：Relu(输入大于0就为本身，小于就为0)，便于模拟  
- 规范化层：用于数据的归一化(可能出现极端数据，如10000000000，-1，0.2，1之类的)，一般是求线性回归。
- 加输入的部分有时候会用到1*1卷积核，为了统一通道数（通道:可以理解为需要提取的特征数，程序会自动根据输入的通道数，提取特征）
- ##3.程序编写如model.py所示，下面几行代码是测试用的
```
import torch
from torch import nn
from torchsummary import summary
import numpy as np
flag = torch.cuda.is_available()
print(flag)

ngpu = 1
device = torch.device('cuda:0' if flag and ngpu > 0 else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
#print(torch.rand(3, 3).cuda())

cuda_version = torch.version.cuda
print("CUDA version:", cuda_version)

# cudnn_version = torch.backends.cudnn.version()
# print("CUDNN version:", cudnn_version)
class Residul(nn.Module):
    def __init__(self,inputchannels,outchannels,use_1=1,strides=1):
        super(Residul, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=inputchannels,out_channels=outchannels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(in_channels=outchannels,out_channels=outchannels,kernel_size=3,padding=1)
        self.batchNormalizer_1 = nn.BatchNorm2d(outchannels)
        self.batchNormalizer_2 = nn.BatchNorm2d(outchannels)
        if use_1==1:
            self.conv3=nn.Conv2d(in_channels=inputchannels,out_channels=outchannels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
    def forward(self,inputs):
        output = self.conv1(inputs)
        output = self.batchNormalizer_1(output)
        output = self.Relu(output)
        output = self.conv2(output)
        output = self.batchNormalizer_2(output)
        if self.conv3 is not None:
            output = self.Relu(output+self.conv3(inputs))
        else:
            output = self.Relu(output+inputs)
        return output

#以下是测试用的
model_a=Residul(inputchannels=3,outchannels=4,use_1=1,strides=1)
model_a=model_a.to(device)
output_a=model_a(torch.randn(1,3,64,64).cuda())
print(output_a.shape)
```
