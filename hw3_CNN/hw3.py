import time
import os
import torch as tc
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torch.utils.data import dataset, dataloader  # 用于包装data，方便training和testing


# read pics

def read_file(path, label):
    img_dir = sorted(os.listdir(path))
    LEN = len(img_dir) #len(img_dir)
    x = np.zeros((LEN, 128, 128, 3), 'uint8')   # 图片必须是uint8
    y = np.zeros((LEN, ), 'uint8')
    for i, file in enumerate(img_dir):
        if i >= LEN:
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
dir_path = './food-11'
train_x, train_y = read_file(os.path.join(dir_path, "training"), True)
valid_x, valid_y = read_file(os.path.join(dir_path, "validation"), True)
# test_x = read_file('food/testing',False)
print(f'read train file over. Time {round(time.time() - s_time, 2)} s')
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


class ImgDataset(dataset.Dataset):  # 继承父类dataset
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = tc.LongTensor(y)  # y 应该为Long Tensor
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
train_set = ImgDataset(train_x, train_y, train_transform) # 顺便进行数据的transform，对于数据进行预处理
valid_set = ImgDataset(valid_x, valid_y, test_transform)
train_loader = dataloader.DataLoader(train_set, batch_size, True) # 设置batch_size对数据进行小批量训练。并且打乱顺序。
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
            nn.BatchNorm2d(64), # norm2d的输入为特征数量，并进行标准化，这里可以设置momentum和eps
            nn.ReLU(),  # 进行ReLU函数转换
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
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x) # 进行CNN转换
        out = out.view(out.size()[0], -1)   # 总感觉这句话没什么用
        return self.fc(out) # 输入网络，有四层网络：输入层，隐藏层1，隐藏层2，输出层。最后输出11维度的向量（有11种类别的食品）。


model = Classifier().cuda()   # 采用GPU模式，构建模型实例
loss = nn.CrossEntropyLoss()  # 因为属于分类，因此定义交叉熵损失函数
# optimizer = tc.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))   # Use adam optimizer, lr是指learning rate
optimizer = tc.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
num_epoch = 90

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = val_acc = 0
    train_loss = val_loss = 0
    model.train()   # 告诉模型开始训练了，启用batch标准化和dropout
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 清除原先的导数数据，进行新一轮batch的求导
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].to('cuda',dtype=tc.long)) # Loss 这里需要做类型转换
        batch_loss.backward() # back propagation
        optimizer.step()  # 设置一个step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        # 得到的train_pred每一行是对11中类别的可能行权值，选取每一行中最大的结果的index，argmax返回的是index
        train_loss += batch_loss.item()

    model.eval()
    # 全称 evaluation，不启用batchNormalization和dropout，否则网络的权值会发生改变
    # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval()时，
    # 框架会自动把BatchNormalization和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，
    # 很容易就会被BatchNormalization层导致生成图片颜色失真极大！！！！！！
    with tc.no_grad():  # 停止求导
        for i, data in enumerate(valid_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())  # 注意这里的label需要转为long类型，并且只有GPU和GPU之间的变量才能进行运算

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 打印结果
        print(f'[{epoch+1}/{num_epoch}]: Time: {round(time.time() - epoch_start_time,2)}s, Train acc/loss:{round(train_acc / len(train_set), 6) * 100}% / {round(train_loss / len(train_set), 6) * 100}%, Validation acc/loss:{round(val_acc / len(valid_set), 6) * 100}% / {round(val_loss / len(valid_set), 6) * 100}%')

# 使用最好的参数
train_val_x = np.concatenate((train_x, valid_x), axis=0)
train_val_y = np.concatenate((train_y, valid_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_dataloader = dataloader.DataLoader(train_val_set, batch_size, True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()
# optimizer = tc.optim.Adam(model_best.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = tc.optim.SGD(model_best.parameters(), lr = 0.001, momentum=0.9)
num_epoch = 90
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc, train_loss = 0, 0

    model_best.train()
    for i, data in enumerate(train_val_dataloader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    print(f'[{epoch+1}/{num_epoch}]: Time: {time.time() - epoch_start_time}, Train acc/loss:{round(train_acc / len(train_val_set), 2) * 100}% / {round(train_loss / len(train_val_set), 2) * 100}%')


tc.save(model_best,"model_best.torch")  # 保存模型

test_x = read_file(os.path.join(dir_path, 'testing'), False)
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = dataloader.DataLoader(test_set, batch_size, False)
model_best.eval()

predication = []
with tc.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            predication.append(y)

rs = pd.DataFrame(predication,columns=['label'])
rs.to_csv('ans.csv')

print(f'Cost time: {round(time.time() - s_time, 2)} s')
