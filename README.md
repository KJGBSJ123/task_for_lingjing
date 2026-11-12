# task_for_lingjing
社团考核用

**2026/11/10**
- ## 学习残差块的构建
- ### 1.ResNet：  
- 最新的卷积神经网络，用于解决训练轮数增加导致准确度不升反降的问题。  
- 方法：只算残差，然后直接加输入值  
- ### 2.残差块的组成：  
- 输入->卷积层1->规范化层1->Relu激活层->卷积层2->规范化层2->(+输入)->Relu激活层  
- 卷积层：用卷积核扫描，并获取信息  
- 激活层：Relu(输入大于0就为本身，小于就为0)，便于模拟  
- 规范化层：用于数据的归一化(可能出现极端数据，如10000000000，-1，0.2，1之类的)，一般是求线性回归。
- 加输入的部分有时候会用到1*1卷积核，为了统一通道数（通道:可以理解为需要提取的特征数，程序会自动根据输入的通道数，提取特征）
- ### 3.程序编写如[ResNet/model.py](ResNet/model.py)所示，下面几行代码是测试用的
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
- **2026/11/11**
- ## 学习搭建ResNet
- 新增代码如下：
```
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
```
- *b1:初始特征提取，池化*
- （池化：取卷积核中最大值，保留最关键特征，比如）
- 1 2 3
- 4 5 6     -->9
- 7 8 9
- *b2-5:卷积层，提取，学习关键信息*
- 这里用到了昨天的残差块
- *b6:展平，线形层分析*


- **2026/11/12**
- ## 训练
- 训练代码如下
```
  import time
import pandas as pd
import torch
import copy
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import numpy as np
from model import ResNet

def train_val_data():
    train_dataset = FashionMNIST(root='./data',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.Resize(size=32),transforms.ToTensor()]))
    train_data,val_data=Data.random_split(train_dataset,[int(len(train_dataset)*0.8),int(len(train_dataset)*0.2)])
    train_loader = Data.DataLoader(dataset=train_data,batch_size=128,shuffle=True,num_workers=0)
    val_loader = Data.DataLoader(dataset=val_data,batch_size=128,shuffle=True,num_workers=0)
    return train_loader,val_loader

def train_model(model,train_loader,val_loader,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)#Adam优化算法
    criterion = nn.CrossEntropyLoss()#交叉熵损失
    model=model.to(device)
    best_model=copy.deepcopy(model.state_dict())#复制当前参数
    best_accuracy=0.0#准确率
    val_loss_all=[]
    train_loss_all=[]#损失列表
    val_acc=[]
    train_acc=[]#准确率列表
    for epoch in range(epochs):
        since = time.time()
        print("-"*10)
        print(f"Epoch {epoch}/{epochs-1}")

        train_loss=0.0
        train_correct=0.0


        val_loss=0.0
        val_correct=0.0

        train_num=0
        val_num=0#样本数量

        for batch_idx, (b_x, b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            model.train()

            output = model(b_x)#模型

            pre_lab=torch.argmax(output,dim=1)#Softmax最大结果
            loss=criterion(output,b_y)

            optimizer.zero_grad()#梯度为0
            loss.backward()#反向传播
            optimizer.step()

            train_loss += loss.item()*b_x.size(0)#取值
            train_correct += torch.sum(pre_lab==b_y.data)
            train_num += b_x.size(0)
        for batch_idx, (b_x, b_y) in enumerate(val_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)

            model.eval()#评估模式

            output = model(b_x)  # 模型

            pre_lab = torch.argmax(output, dim=1)  # Softmax最大结果
            loss = criterion(output, b_y)

            val_loss += loss.item()*b_x.size(0)#取值loss.item()是平均值
            val_correct += torch.sum(pre_lab==b_y.data)
            val_num+=b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        train_acc.append(train_correct.double().item()/train_num)

        val_loss_all.append(val_loss/val_num)
        val_acc.append(val_correct.double().item()/val_num)

        print(f"Train loss: {train_loss_all[-1]:.4f}  Train Acc: {train_acc[-1]:.4f}")
        print(f"Val loss: {val_loss_all[-1]:.4f}  Val Acc: {val_acc[-1]:.4f}")

        if val_acc[-1]>best_accuracy:
            best_accuracy=val_acc[-1]
            best_model=copy.deepcopy(model.state_dict())
        time_use=time.time()-since
        print(f"Time taken:{time_use:.4f}seconds")

        model.load_state_dict(best_model)
    torch.save(model.state_dict(),"Model/best_model.pth")
    train_process = pd.DataFrame(data={"epoch":range(epochs),
                                           "train_loss":train_loss_all,
                                           "train_acc":train_acc,
                                           "val_loss":val_loss_all,
                                           "val_acc":val_acc
                                           })
    return train_process


if __name__=="__main__":
    flag = torch.cuda.is_available()
    print(flag)

    ngpu = 1
    device = torch.device('cuda:0' if flag and ngpu > 0 else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    # print(torch.rand(3, 3).cuda())

    cuda_version = torch.version.cuda
    print("CUDA version:", cuda_version)
    train_loader,val_loader=train_val_data()
    train_model(ResNet(),train_loader,val_loader,20)


  ```
- 解析：
- ### train_val_data():
- 负责加载数据集，将数据集划分为训练集（train_loader）和验证集(val_loader)
- ### train_model(model,train_loader,val_loader,epochs)
- 负责训练的过程：
- 重复epochs轮，对于每轮：
- （1）训练：输入特征（b_x),得到结果与标签（b_y)对比，通过LOSS函数计算损失，optimizer计算梯度，反向传播更新参数
- （2）评估：输入验证集，计算并记录损失与准确率
- 最后，找到并记录最优模型
