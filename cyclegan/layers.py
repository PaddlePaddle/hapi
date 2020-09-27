# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer, Conv2d, BatchNorm, ConvTranspose2d


class ConvBN(Layer):
    """docstring for Conv2d"""

    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            padding=0,
            stddev=0.02,
            norm=True,
            #is_test=False,
            act='leaky_relu',
            relufactor=0.0,
            use_bias=False):
        super(ConvBN, self).__init__()

        pattr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            loc=0.0, scale=stddev))
        self.conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=pattr,
            bias_attr=use_bias)
        if norm:
            self.bn = BatchNorm(
                num_filters,
                param_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(1.0, 0.02)),
                bias_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Constant(0.0)),
                is_test=False,
                trainable_statistics=True)
            #track_running_stats=True)
        self.relufactor = relufactor
        self.norm = norm
        self.act = act

    def forward(self, inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)

        if self.act == 'leaky_relu':
            conv = F.leaky_relu(conv, self.relufactor)
        elif self.act == 'relu':
            conv = F.relu(conv)
        else:
            conv = conv

        return conv


class DeConvBN(Layer):
    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            padding=[0, 0],
            outpadding=[0, 0, 0, 0],
            stddev=0.02,
            act='leaky_relu',
            norm=True,
            #is_test=False,
            relufactor=0.0,
            use_bias=False):
        super(DeConvBN, self).__init__()

        pattr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            loc=0.0, scale=stddev))
        self._deconv = ConvTranspose2d(
            num_channels,
            num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=pattr,
            bias_attr=use_bias)
        if norm:
            self.bn = BatchNorm(
                num_filters,
                param_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(1.0, 0.02)),
                bias_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Constant(0.0)),
                is_test=False,
                trainable_statistics=True)
            #track_running_stats=True)
        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.act = act

    def forward(self, inputs):
        conv = self._deconv(inputs)
        conv = F.pad2d(
            conv, paddings=self.outpadding, mode='constant', pad_value=0.0)

        if self.norm:
            conv = self.bn(conv)

        if self.act == 'leaky_relu':
            conv = F.leaky_relu(conv, self.relufactor)
        elif self.act == 'relu':
            conv = F.relu(conv)
        else:
            conv = conv

        return conv
