### 使用pytorch实现表情分类
#### 版本
	os: unbuntu18.04
	python：
	pytorch：
	opencv：
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

2. 开始构建数据集

```python
class FaceDataset(data.Dataset):
    # 面部数据集
    def __init__(self, path, transform=None, target_transform=None):
        super(FaceDataset, self).__init__()
        # imgs和labels数组包含了图片路径和标签信息
        self.imgs = np.array(imgs)
        self.labels = np.array(labelArray)

        # transform是一个Compose类型，包含对图像的翻转，随机裁剪等操作
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # 通过路径读取图像，index为路径数组的索引
        return face_tensor, label

    def __len__(self):
        return self.imgs.shape[0]
```