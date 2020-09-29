#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import paddle.fluid as fluid
# from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from paddle.static import InputSpec
from paddle.utils.download import get_weights_path_from_url

__all__ = ["TSM_ResNet", "tsm_resnet50"]

# {num_layers: (url, md5)}
pretrain_infos = {
    50: ('https://paddlemodels.bj.bcebos.com/hapi/tsm_resnet50.pdparams',
         '5755dc538e422589f417f7b38d7cc3c7')
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        self._batch_norm = nn.BatchNorm(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'relu':
            y = F.relu(y)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 seg_num=8):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)
        self.shortcut = shortcut
        self.seg_num = seg_num
        self._num_channels_out = int(num_filters * 4)

    def forward(self, inputs):
        shifts = F.temporal_shift(inputs, self.seg_num, 1.0 / 8)
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.elementwise_add(x=short, y=conv2, act="relu")
        return y


class TSM_ResNet(nn.Layer):
    """
    TSM network with ResNet as backbone

    Args:
        num_layers (int): ResNet layer number, only support 50 currently.
            Default 50.
        seg_num (int): segment number of each video sample. Default 8.
        num_classes (int): video class number. Default 400.
    """

    def __init__(self, num_layers=50, seg_num=8, num_classes=400):
        super(TSM_ResNet, self).__init__()

        self.layers = num_layers
        self.seg_num = seg_num
        self.class_dim = num_classes

        if self.layers == 50:
            depth = [3, 4, 6, 3]
        else:
            raise NotImplementedError
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        # self.pool2d_max = Pool2D(
        #     pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        # self.pool2d_max = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.bottleneck_block_list = []
        num_channels = 64

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        seg_num=self.seg_num))
                num_channels = int(bottleneck_block._num_channels_out)
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        # self.pool2d_avg = Pool2D(
        #     pool_size=7, pool_type='avg', global_pooling=True)
        # self.pool2d_avg = nn.AdaptiveAvgPool2d(output_size=1)

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = nn.Linear(
            2048,
            # self.class_dim)
            self.class_dim,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2.0, regularizer=paddle.regularizer.L2Decay(0.)))


    def forward(self, inputs):
        y = paddle.reshape(
            inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
        y = self.conv(y)
        y = F.max_pool2d(y, kernel_size=3, stride=2, padding=1)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = F.adaptive_avg_pool2d(y, output_size=1)
        y = F.dropout(y, p=0.5)
        y = paddle.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = paddle.reduce_mean(y, dim=1)
        y = paddle.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        return y


def _tsm_resnet(num_layers, seg_num=8, num_classes=400, pretrained=True):
    net = TSM_ResNet(num_layers, seg_num, num_classes)
    if pretrained:
        assert num_layers in pretrain_infos.keys(), \
                "TSM-ResNet{} do not have pretrained weights now, " \
                "pretrained should be set as False".format(num_layers)
        weight_path = get_weights_path_from_url(*(pretrain_infos[num_layers]))
        assert weight_path.endswith('.pdparams'), \
                "suffix of weight must be .pdparams"
        # # weight_dict, _ = fluid.load_dygraph(weight_path)
        # # model.set_dict(weight_dict)
        # model.load(weight_path)
        params = paddle.load(weight_path)
        net.load_dict(params)

    inputs = [InputSpec([None, 8, 3, 224, 224], 'float32', name='image')]
    labels = [InputSpec([None, 1], 'int64', name='label')]
    model = paddle.Model(net, inputs, labels)
    return model


def tsm_resnet50(seg_num=8, num_classes=400, pretrained=True):
    """TSM model with 50-layer ResNet as backbone
    
    Args:
        seg_num (int): segment number of each video sample. Default 8.
        num_classes (int): video class number. Default 400.
        pretrained (bool): If True, returns a model with pre-trained model
            on COCO, default True
    """
    return _tsm_resnet(50, seg_num, num_classes, pretrained)

