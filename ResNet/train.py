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

