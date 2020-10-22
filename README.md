# 简介

飞桨框架2.0全新推出高层API，是对飞桨API的进一步封装与升级，提供了更加简洁易用的API，进一步提升了飞桨的易学易用性，并增强飞桨的功能。

飞桨高层API面向从深度学习小白到资深开发者的所有人群，对于AI初学者来说，使用高层API可以简单快速的构建深度学习项目，对于资深开发者来说，可以快速完成算法迭代。

飞桨高层API具有以下特点：

- 易学易用: 高层API是对普通动态图API的进一步封装和优化，同时保持与普通API的兼容性，高层API使用更加易学易用，同样的实现使用高层API可以节省大量的代码。
- 低代码开发: 使用飞桨高层API的一个明显特点是，用户可编程代码量大大缩减。
- 动静转换: 高层API支持动静转换，用户只需要改一行代码即可实现将动态图代码在静态图模式下训练，既方便用户使用动态图调试模型，又提升了模型训练效率。

在功能增强与使用方式上，高层API有以下升级：

- 模型训练方式升级: 高层API中封装了Model类，继承了Model类的神经网络可以仅用几行代码完成模型的训练。
- 新增图像处理模块transform: 飞桨新增了图像预处理模块，其中包含数十种数据处理函数，基本涵盖了常用的数据处理、数据增强方法。
- 提供常用的神经网络模型可供调用: 高层API中集成了计算机视觉领域和自然语言处理领域常用模型，包括但不限于mobilenet、resnet、yolov3、cyclegan、bert、transformer、seq2seq等等。同时发布了对应模型的预训练模型，用户可以直接使用这些模型或者在此基础上完成二次开发。


![](./image/hapi_gif.gif)


## 目录

* [特性](#1)
* [快速使用](#2)
* [新增功能](#3)
* [使用示例](#4)


## <h2 id="1">特性</h2>

### 易学易用

高层API基于飞桨动态图实现，兼容飞桨动态图的所有功能，既秉承了动态图易学、易用、易调试的特点，又对飞桨的动态图做了进一步的封装与优化。

### 低代码开发

相比较与动态图的算法实现，使用高层API实现的算法可编程代码量更少，原始的动态图训练代码需要20多行代码才能完成模型的训练，使用高层API后，仅用8行代码即可实现相同的功能。

使用普通API与高层API实现手写字符识别对比如下图，左边是普通动态图API的实现，右边是使用高层API的实现，可以明显发现，使用高层API的代码量更少。

![](./image/new_hapi.png)


### 动静统一

高层API中实现了动静统一，用户无需感知到静态图、动态图的区别，只需要改一行代码即可实现将动态图代码在静态图模式下训练。动态图更方便调试模型，静态图的训练方式训练效率更高。

高层API默认训练方式和主框架保持一致，采用动态图的训练方式，我们可以使用`paddle.disable_static()`来开启动态图训练模式，用`paddle.enable_static()`开启静态图模式训练。

```python
# 一行代码切换动态图训练模式
## 动态图训练模式
paddle.disable_static()
## 静态图训练模式
# paddle.enable_static()

# 设置训练设备环境
paddle.set_device('gpu')

# 声明网络结构
model = paddle.Model(Mnist())
# 定义优化器
optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
# 调用prepare() 完成训练的配置
model.prepare(optimizer, CrossEntropy(), Accuracy())
# 调用 fit()，启动模型的训练
model.fit(train_dataset, val_dataset, batch_size=100, epochs=1, log_freq=100, save_dir="./output/")
```

## <h3 id="2">快速使用</h3>

以mnist手写字符识别为例，介绍飞桨高层API的使用方式。

### 1. 搭建网络结构

使用高层API组建网络与动态图的组网方式完全相同，继承`paddle.nn.Layer`来定义网络结构即可。

高层API组网方式如下
```python
import paddle

# 设置执行环境为GPU
paddle.set_device('gpu')
# 使用动态图训练方式
paddle.disable_static()

class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()
        self.fc = paddle.nn.Linear(input_dim=784, output_dim=10)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)

        return outputs
```

### 2. 训练准备

在开始训练前，需要定义优化器、损失函数、度量函数，准备数据等等。这些过程均可以在高层API Model类中的prepare函数中完成。

```python
# 定义输入数据格式
inputs = [Input([None, 784], 'float32', name='image')]
labels = [Input([None, 1], 'int64', name='label')]

# 声明网络结构
model = paddle.Model(Mnist())
optimizer = paddle.optimizer.SGD(learning_rate=0.001,
                                 parameters=model.parameters())
# 使用高层API，prepare() 完成训练的配置
model.prepare(optimizer,
              paddle.nn.CrossEntropy(),
              paddle.metricAccuracy())
```

### 3. 启动训练

使用高层API完成训练迭代过程时，使用一行代码即可构建双层循环程序，去控制训练的轮数和数据读取过程。

```python
from paddle.vision.datasets import MNIST as MnistDataset
# 定义数据读取器
train_dataset = MnistDataset(mode='train')
val_dataset = MnistDataset(mode='test')
# 启动训练
model.fit(train_dataset, val_dataset, batch_size=100, epochs=10, log_freq=100, save_dir="./output/")
```

高层API中通过fit函数完成训练的循环过程，只需要设置训练的数据读取器、batchsize大小，迭代的轮数epoch、训练日志打印频率log_freq，保存模型的路径即可。

## <h4 id="3">新增功能</h4>

除了使用高层API实现一行代码启动训练外，还新增了以下功能：
- paddle.vision.transforms   图像数据增强模块
- paddle.vision.models  模型调用模块

### transforms
paddle.vision.transforms。图像预处理模块transforms包括一系列的图像增强与图像处理实现，对处理计算机视觉相关的任务有很大帮助。

下表中列出Transforms支持的数据处理和数据增强API，如下所示：

| transform的数据处理实现  | 函数功能 |
| :--------   | :-----   |
|  Compose  | 组合多种数据变换 |
| BatchCompose | 用于处理批数据的预处理接口组合 |
|  Resize  | 将图像转换为固定大小 |
| RandomResizedCrop  |  根据输入比例对图像做随机剪切，然后resize到指定大小   |
|  CenterCrop  | 以图像的中心为中心对图像做剪切 |
|  CenterCropResize  | 对图像做padding，padding后的图像做centercrop，然后resize到指定大小|
|  RandomHorizontalFlip |  随机对图像做水平翻转   |
|  RandomVerticalFlip |  随机对图像做垂直翻转   |
| RandomCrop | 在随机位置裁剪输入的图像 |
| RandomErasing | 随机选择图像中的一个矩形区域并将其像素删除 |
| RandomRotate | 按角度旋转图像 |
|  Permute |  将数据的的维度换位   |
|  Normalize |   用指定的均值和标准差对数据做归一化  |
| GaussianNoise  |  给数据增加高斯噪声   |
|  BrightnessTransform |  调整输入图像的亮度   |
|  SaturationTransform |  调整输入图像的饱和度   |
|  ContrastTransform |  调整输入图像的对比度   |
| HueTransform  |   调整图像的色调  |
|  ColorJitter |  随机调整图像的亮度、饱和度、对比度、和色调|
| Grayscale | 将图像转换为灰度 |
| Pad | 使用特定的填充模式和填充值来对输入图像进行填充 |

使用方法如下：
```python
from paddle.vision import transforms
import cv2

img_path = "./output/sample.jpg"
img = cv2.imread(img_path)

# 使用Compose 将可以将多个数据增强函数组合在一起
trans_funcs = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.BrightnessTransform(0.2)])
label = None
img_processed, label = trans_funcs(img, label)
```

上述代码的效果图如下：

![](./image/hapi_transform.png)


### paddle.vision.models

paddle.vision.models中包含了高层API对常用模型的封装，包括ResNet、VGG、MobileNet、LeNet等。使用这些现有的模型，可以快速的完成神经网络的训练、finetune等。

使用paddle.vision中的模型可以简单快速的构建一个深度学习任务，比如13代码即可实现resnet在Cifar10数据集上的训练：

```python
from paddle.vision.models import resnet50
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropy
from paddle.metirc import Accuracy


# 调用resnet50模型
model = resnet50(pretrained=False)
# 使用Cifar10数据集
train_dataset = Cifar10(mode='train')
val_dataset = Cifar10(mode='test')
# 定义优化器
optimizer = Momentum(learning_rate=0.01,
                     momentum=0.9,
                     weight_decay=L2Decay(1e-4),
                     parameters=model.parameters())
# 进行训练前准备
model.prepare(optimizer, CrossEntropy(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(train_dataset,
          val_dataset,
          epochs=50,
          batch_size=64,
          save_dir="./output",
          num_workers=8)
```



## <h5 id="4">更多使用示例</h5>

更多的高层API使用示例请参考：
- [bert](https://github.com/PaddlePaddle/hapi/tree/master/bert)
- [image classification](https://github.com/PaddlePaddle/hapi/tree/master/image_classification)
- [BMN](https://github.com/PaddlePaddle/hapi/tree/master/bmn)
- [CycleGAN](https://github.com/PaddlePaddle/hapi/tree/master/cyclegan)
- [ocr](https://github.com/PaddlePaddle/hapi/tree/master/ocr)
- [TSM](https://github.com/PaddlePaddle/hapi/tree/master/tsm)
- [yolov3](https://github.com/PaddlePaddle/hapi/tree/master/yolov3)
- [transformer](https://github.com/PaddlePaddle/hapi/tree/master/transformer)
- [seq2seq](https://github.com/PaddlePaddle/hapi/tree/master/seq2seq)
- [style-transfer](https://github.com/PaddlePaddle/hapi/tree/master/style-transfer)
