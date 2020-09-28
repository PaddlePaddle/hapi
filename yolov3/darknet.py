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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.utils.download import get_weights_path_from_url

__all__ = ['DarkNet', 'darknet53']

# {depth: (url, md5)}
pretrain_infos = {
    53: ('https://paddlemodels.bj.bcebos.com/hapi/darknet53.pdparams',
         '0dc32d7b7d1d3ee0406fc2b94eb660ff')
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '.conv.weights'),
            bias_attr=False)
        bn_name = name + '.bn'
        self.batch_norm = nn.BatchNorm2d(
            ch_out,
            weight_attr=ParamAttr(
                name=bn_name + '.scale', regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                name=bn_name + '.offset', regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(out, 0.1)
        return out


class DownSample(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 name=None):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            name=name)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, name=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            name=name + '.0')
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            name=name + '.1')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.add(x=inputs, y=conv2)
        return out


class Blocks(nn.Layer):
    def __init__(self, ch_in, ch_out, count, name=None):
        super(Blocks, self).__init__()

        self.basicblock0 = BasicBlock(ch_in, ch_out, name=name + '.0')
        self.res_out_list = []
        for i in range(1, count):
            block_name = '{}.{}'.format(name, i)
            res_out = self.add_sublayer(
                block_name, BasicBlock(
                    ch_out * 2, ch_out, name=block_name))
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet(nn.Layer):
    def __init__(self,
                 depth=53,
                 freeze_at=-1,
                 return_idx=[2, 3, 4],
                 num_stages=5):
        super(DarkNet, self).__init__()
        self.depth = depth
        self.freeze_at = freeze_at
        self.return_idx = return_idx
        self.num_stages = num_stages
        self.stages = DarkNet_cfg[self.depth][0:num_stages]

        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            name='yolo_input')

        self.downsample0 = DownSample(
            ch_in=32, ch_out=32 * 2, name='yolo_input.downsample')

        self.darknet_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]
        for i, stage in enumerate(self.stages):
            name = 'stage_{}'.format(i)
            conv_block = self.add_sublayer(
                name, Blocks(
                    int(ch_in[i]), 32 * (2**i), stage, name=name))
            self.darknet_conv_block_list.append(conv_block)
        for i in range(num_stages - 1):
            down_name = 'stage_{}.downsample'.format(i)
            downsample = self.add_sublayer(
                down_name,
                DownSample(
                    ch_in=32 * (2**(i + 1)),
                    ch_out=32 * (2**(i + 2)),
                    name=down_name))
            self.downsample_list.append(downsample)

    def forward(self, x):
        out = self.conv0(x)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet_conv_block_list):
            out = conv_block_i(out)
            if i == self.freeze_at:
                out.stop_gradient = True
            if i in self.return_idx:
                blocks.append(out)
            if i < self.num_stages - 1:
                out = self.downsample_list[i](out)
        return blocks


def _darknet(depth=53, pretrained=True):
    model = DarkNet(depth)
    if pretrained:
        assert depth in pretrain_infos.keys(), \
                "DarkNet{} do not have pretrained weights now, " \
                "pretrained should be set as False".format(num_layers)
        weight_path = get_weights_path_from_url(*(pretrain_infos[depth]))
        assert weight_path.endswith('.pdparams'), \
                "suffix of weight must be .pdparams"
        params = paddle.load(weight_path)
        model.load_dict(params)
    return model


def darknet53(pretrained=True):
    """DarkNet 53-layer model
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet,
            default True.
    """
    return _darknet(53, pretrained)

if __name__ == "__main__":
    d = darknet53(pretrained=True)
    # paddle.save(d.state_dict(), 'd.params')
    # d.load_dict(paddle.load('./new_darknet53.pdparams'))
