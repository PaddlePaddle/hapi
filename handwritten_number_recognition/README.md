# MNIST
当我们学习编程的时候，编写的第一个程序一般是实现打印"Hello World"。而机器学习（或深度学习）的入门教程，一般都是 MNIST 数据库上的手写识别问题。原因是手写识别属于典型的图像分类问题，比较简单，同时MNIST数据集也很完备。
本页将介绍如何使用PaddlePaddle高级API(hapi)实现MNIST，包括[安装](#installation)、[训练](#training-a-model)、[输出](#log)、[参数保存](#save)、[模型评估](#evaluation)。


## 安装

在当前目录下运行样例代码需要PadddlePaddle的v2.0.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据安装文档中的说明来更新PaddlePaddle。

## 训练
可以通过如下的方式启动训练：
```
python mnist.py
```
上面的方式默认使用的静态图模式，切换动态图模式训练可以加```--dynamic```
```
python mnist.py --dynamic
```
多卡进行模型训练，启动训练的方式：
```
python -m paddle.distributed.launch mnist.py
```

## 输出
执行训练开始后，将得到类似如下的输出。
```
Epoch 1/10
step  10/469 - loss: 2.4547 - acc_top1: 0.1273 - acc_top2: 0.2305 - 94ms/step
step  20/469 - loss: 1.2578 - acc_top1: 0.3063 - acc_top2: 0.4316 - 48ms/step
step  30/469 - loss: 0.7918 - acc_top1: 0.4344 - acc_top2: 0.5638 - 33ms/step
step  40/469 - loss: 0.6947 - acc_top1: 0.5148 - acc_top2: 0.6412 - 25ms/step
step  50/469 - loss: 0.5452 - acc_top1: 0.5731 - acc_top2: 0.6959 - 20ms/step
step  60/469 - loss: 0.4184 - acc_top1: 0.6133 - acc_top2: 0.7314 - 17ms/step
step  70/469 - loss: 0.5143 - acc_top1: 0.6423 - acc_top2: 0.7595 - 15ms/step
step  80/469 - loss: 0.5688 - acc_top1: 0.6658 - acc_top2: 0.7808 - 13ms/step
...
```

## 参数保存
训练好的模型默认会保存在```mnist_checkpoint/```文件加下，可以通过```--output-dir```命令来指定你想要保存的文件夹位置。


## 模型评估
执行如下命令进行评估，```--resume```后面指定训练好的模型路径
```
python mnist.py --resume mnist_checkpoint/final.pdparams --eval-only
```
切换动态图模式评估：
```
python mnist.py --resume mnist_checkpoint/final.pdparams --eval-only --dynamic
```
多卡评估
```
python -m paddle.distributed.launch mnist.py --resume mnist_checkpoint/final.pdparams --eval-only
```
