# -*- coding: utf-8 -*
import os
import shutil
import torch
from torch.utils import data
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, models, transforms

root = ''
Train_txt = root + "CK+_Trai分类n.txt"
Val_txt = root + "CK+_Eval.txt"


class FaceCNN(nn.Module):

    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积
        self.conv1 = nn.Sequential(

            # input_image(batch_size,1,112,112),output(batch_size,64,112,112)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 归一化
            nn.BatchNorm2d(num_features=64),
            # 激活函数
            nn.RReLU(inplace=True),
            # 池化
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二次卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三次卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv1.apply(gaussian_weights_init)

        self.conv2.apply(gaussian_weights_init)

        self.conv3.apply(gaussian_weights_init)
        # print(self.conv1, self.conv2, self.conv3)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*14*14, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7)
        )

        # 向前传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        # 数据展平
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


# 分离出训练集和验证集数据
def split_train_val(txt_path, mode):

    # open `.txt`file
    f = open(txt_path, 'r')
    while 1:
        line = f.readline()
        if line:
            x = line.split()
            # image path
            oldImgPath = str('CK+last5-align/CK+last5-align' + '/' + x[1] + '/' + str(x[0].split('/')[0]) + '/')
            newImgPath = str(mode + '/')
            filelist = os.listdir(oldImgPath)
            for filename in filelist:
                filepath = os.path.join(oldImgPath, filename)
                newfilepath = newImgPath + x[1] + '_' + filename
                print(newfilepath)
                # copy image path
                shutil.copy(filepath, newfilepath)

            y = str(x[0]).replace('/', '_')
            print(str(y), oldImgPath, newImgPath)
        if not line:
            break
    f.close()
    return 0


# 构建dataset数据集
class FaceDataset(data.Dataset):
    # 面部数据集
    def __init__(self, path, transform=None, target_transform=None):
        super(FaceDataset, self).__init__()
        # path为train或val的地址
        self.path = path
        filelist = os.listdir(self.path)
        imgs = []
        labelArray = []
        for filename in filelist:
            label = filename.split('_')
            # 将标签转化为0-6的数字，对照字典如下：
            labellist = {'Angry': 0,
                         'Contempt': 1,
                         'Disgust': 2,
                         'Fear': 3,
                         'Happy': 4,
                         'Sad': 5,
                         'Surprise': 6}
            label = labellist[label[0]]
            labelArray.append(label)
            img_path = self.path + filename
            imgs.append(img_path)

        # imgs和labels数组包含了图片路径和标签信息
        self.imgs = np.array(imgs)
        self.labels = np.array(labelArray)

        # transform是一个Compose类型，包含对图像的翻转，随机裁剪等操作
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # 通过路径读取图像，index为路径数组的索引
        face = cv2.imread(self.imgs[index])
        # 单通道灰度图
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 像素值标准化
        normalized = gray.reshape(1, 112, 112) / 255.0
        # 转化为tensor形式
        face_tensor = torch.from_numpy(normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')

        label = self.labels[index]

        return face_tensor, label

    def __len__(self):
        return self.imgs.shape[0]


# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
 
    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1
 
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
 
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
 
        #shortcut
        self.shortcut = nn.Sequential()
 
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
 
class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
 
        self.shortcut = nn.Sequential()
 
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNet(nn.Module):
 
    def __init__(self, block, num_block, num_classes=7):
        super().__init__()
 
        self.in_channels = 64
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1,padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """
 
        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
 
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
 
        return output 
 
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])
 
def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])
 
def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])
 
def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])
 
def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


# 训练函数
def train(TrainDataset, ValDataset, batch_size, epochs, lr, wt_decay):

    train_loader = data.DataLoader(TrainDataset, batch_size)

    # model = FaceCNN()
    model = resnet34()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wt_decay)
    # 训练
    loss = []
    val = []
    for epoch in range(epochs):
        loss_rate = 0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model.forward(images)
            # print(output.size())
            loss_rate = loss_function(output, labels)
            loss_rate.backward()
            optimizer.step()
        

        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        loss.append(loss_rate.item())
        
        
        if epoch % 5 == 0:
            model.eval()
            acc_train = validate(model, TrainDataset, batch_size)
            acc_val = validate(model, ValDataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch + 1), acc_val)
            val.append(acc_val)

    # 将loss写入文件，对比保存
    f1 = open('count.txt', 'a')
    f1.write('resnet34 = ')
    f1.write(str(loss))
    f1.write('\n') 
    f1.write('resnet34_val = ')  
    f1.write(str(val))
    f1.write('\n')      

    return model


train_dataset = FaceDataset(path='train/')
val_dataset = FaceDataset(path='val/')
model = train(train_dataset, val_dataset, batch_size=128, epochs=36, lr=0.08, wt_decay=0)
torch.save(model, 'model_net1.pkl')