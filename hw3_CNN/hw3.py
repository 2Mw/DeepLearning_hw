import time
import os
from numpy.core.arrayprint import printoptions
import torch as tc
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torch.utils.data import dataset, dataloader  # 用于包装data，方便training和testing

# read pics
pic = cv2.imread('food/training/0_0.jpg')


def read_file(path, label):
    img_dir = sorted(os.listdir(path))
    x = np.zeros((len(img_dir), 128, 128, 3), np.int)
    y = np.zeros((len(img_dir)), np.int)
    for i, file in enumerate(img_dir):
        if i > 300:
            break
        img = cv2.imread(f'{path}/{file}')
        x[i, :, :, :] = cv2.resize(img, (128, 128))  # 调整图片大小
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return x


s_time = time.time()
train_x, train_y = read_file('food/training', True)
valid_x, valid_y = read_file('food/validation', True)
# test_x = read_file('food/testing',False)
print(f'read train file over. Time {round(time.time() - s_time), 2} s')
print(f"Train len: {len(train_x)}")

train_transform = transforms.Compose([  # 一系列处理图片使用Compose
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机水平旋转
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor()  # 图片转为tensor
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


class ImgDataset(dataset):  # 继承父类dataset
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            y = tc.LongTensor(y)  # y 应该为Long Tensor
        self.transform = transform

    def __len__(self):  # 实现函数__len__
        return len(self.x)

    def __getitem__(self, index):  # 实现函数__getitem__
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            return X, self.y[index]
        else:
            return X


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
valid_set = ImgDataset(valid_x, valid_y, test_transform)
train_loader = dataloader.DataLoader(train_set, batch_size, True)
valid_loader = dataloader.DataLoader(valid_set, batch_size, False)


# 搭建模型

class Classifier(nn.Module):    # 继承父类nn.Module
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # 原图输入维度 (128, 128, 3)
        self.cnn = nn.Sequential(
            # 输入为彩色图片，输出通道64 ，3X3的卷积核，1步长， 填充长度为1=>为了凑128，不然126不容易解决
            nn.Conv2d(3, 64, 3, 1, 1),  # (128, 128, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # (64, 64, 64)

            # 上一个conv + maxpool的输出为这次的输入
            nn.Conv2d(64, 128, 3, 1, 1),  # (64, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # (32, 32, 128)

            nn.Conv2d(128, 256, 3, 1, 1),  # (32, 32, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # (16, 16, 256)

            nn.Conv2d(256, 512, 3, 1, 1),  # (16, 16, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # (8, 8, 512)

            nn.Conv2d(512, 512, 3, 1, 1),  # (8, 8, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # (4, 4, 512)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


# start train

model = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.parameters(), lr = 0.001)   # Use adam optimizer
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = val_acc = 0
    train_loss = val_loss = 0
    model.train()   # 告诉模型开始训练了，启用batch标准化和dropout
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()    # 全称 evaluation，不启用batchNormalization和dropout，否则网络的权值会发生改变
    # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval()时，
    # 框架会自动把BatchNormalization和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，
    # 很容易就会被BatchNormalization层导致生成图片颜色失真极大！！！！！！


print(f'Cost time: {round(time.time() - s_time, 2)} s')
