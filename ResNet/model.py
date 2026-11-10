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