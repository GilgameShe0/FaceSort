### 使用pytorch实现表情分类

#### 版本

- os: 18.04.1-Ubuntu
- python: Python 3.7.3
- opencv: 4.1.0
- pytorch：1.1.0

####  流程

- 制作数据集
- CNN网络定义
- 送入CNN网络训练
- 优化

#### 代码分析

##### 主要导入包:

os——读取复制移动文件夹
shutil——复制文件夹
torch.untils.data——训练集数据Dataset,DataLoader
numpy——数组处理
cv2——读取图片
torch.nn——继承module
optim——优化器

##### 详细流程

###### 制作数据集

1. 分离train和val数据

建立train、val两个文件夹，分离出`CK+last5-align`文件夹中的训练和验证数据集，打开`CK+_Train.txt`和`CK+_Eval.txt`两个文件，数据格式为`S011/004 Angry`，图片命名格式为`S010_004_00000015.png`，拼写出正确的图片路径把图片按照txt文档分别复制至train、val文件夹，并命名为`原名_标签.png`格式（为了更好地提取出图片的标签）

2. 在pytorch下创建数据集
    
    2.1 使用Dataset数据集
    ```python
    class FaceDataset(data.Dataset):
        # 面部数据集
        def __init__(self, path, transform=None, target_transform=None):
            super(FaceDataset, self).__init__()
            # imgs和labels数组包含了图片路径和标签信息
            self.imgs = np.array(imgs)
            self.labels = np.array(labelArray)

        def __getitem__(self, index):

            # 通过路径读取图像，index为路径数组的索引
            return face_tensor, label

        def __len__(self):
            return self.imgs.shape[0]
    ```
    2.2 创建data-label对照表，FaceDataset继承了Dataset类，在`__init__`中初始化了图片的路径和标签信息，按顺序imgs[0],imgs[1]，对应的label[0],label[1]......

    ```python
    # imgs和labels数组包含了图片路径和标签信息
    self.imgs = np.array(imgs)
    self.labels = np.array(labelArray)

    # transform是一个Compose类型，包含对图像的翻转，随机裁剪等操作
    self.transform = transform
    self.target_transform = target_transform
    ```

    2.3 使用opencv读取图像灰度图,将像素值标准化到[0,1],并返回Tensor形式
        
    ```python
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
    ```
    2.4 数据集的使用
    创建实例：
    ```python
    train_dataset = FaceDataset(path='train/')
    ```
    从train_loader中直接加载出数据和label，每次都会加载出一个批次（batch_size）的数据和label。
    ```python
    train_loader = data.DataLoader(train_dataset, batch_size)
    for images, labels in train_loader:
        #通过images和labels训练模型
    ```

3. CNN模型搭建
    输入的每张图片的大小为[1,112,112]
    3.1 Model B的模型算法
    第一次使用github上开源的算法：
    ```python
    class FaceCNN(nn.Module):

        def __init__(self):
            super(FaceCNN, self).__init__()

            # 第一次卷积
        self.conv1 = nn.Sequential(

            # input_image(batch_size,1,112,112),output(batch_size,64,56,56)
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
            # input_image(batch_size,64,56,56),output(batch_size,128,28,28)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 第三次卷积
        self.conv3 = nn.Sequential(
            # input_image(batch_size,128,28,28),output(batch_size,256,14,14)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(2, 2)
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
    ```
    3.2 使用预训练的Resnet模型
        分别使用不同模型Resnet18，Resnet34，Resnet50，Resnet101，Resnet152
        因为本任务是7分类模型，将Resnet的最后的输出改为7


4. 训练模型
    4.1 选择模型
    ```python
    model = FaceCNN()
    ```
    ```python
    model = resnet18()
    ```
    采用SGD优化器优化训练模型
    ```python
     optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
     ```
    4.2 逐轮训练
    ```python
    # 训练
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
    ```
    4.3 模型的保存与加载
    ```python
    # 模型保存,保存整个模型
    torch.save(model, 'model_net1.pkl')
    ```
    ```python
    # 模型加载
    model_parm =  'model_net1.pkl'
    model = torch.load(net_parm)
    ```
5. 结果分析

    batch_size=128, 
    epochs=36, 
    lr=0.1

    1.使用`model B`训练结果：
    
    
    
    
    
    
    
    
    

