import torch
from torch import nn
from torchsummary import summary
import numpy as np


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

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.b1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.b2=nn.Sequential(
            Residul(64,64,use_1=1),
            Residul(64,64,use_1=1),
        )
        self.b3 = nn.Sequential(
            Residul(64, 128, use_1=1,strides=2),
            Residul(128, 128, use_1=1),
        )
        self.b4 = nn.Sequential(
            Residul(128, 256, use_1=1,strides=2),
            Residul(256, 256, use_1=1),
        )
        self.b5 = nn.Sequential(
            Residul(256, 512, use_1=1,strides=2),
            Residul(512, 512, use_1=1),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,10),
        )
    def forward(self,inputs):
        output = self.b1(inputs)
        output = self.b2(output)
        output = self.b3(output)
        output = self.b4(output)
        output = self.b5(output)
        output = self.b6(output)
        return output
#以下是测试用的
'''
model_a=ResNet(Residul)
model_a=model_a.to(device)
output_a=model_a(torch.randn(1,3,64,64).cuda())
print(output_a.shape)
'''
