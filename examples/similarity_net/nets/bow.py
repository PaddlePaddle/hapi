#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
bow class
"""
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Layer, Embedding
from paddle.incubate.hapi.model import Model


#1. define BOWEncoder
class BOWEncoder(Layer):
    """
    simple BOWEncoder for simnet
    """

    def __init__(self, dict_size, bow_dim, emb_dim, padding_idx):
        super(BOWEncoder, self).__init__()
        self.dict_size = dict_size
        self.bow_dim = bow_dim
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.emb_layer = Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=fluid.ParamAttr(
                name='emb', initializer=fluid.initializer.Xavier()))

    def forward(self, input):
        emb = self.emb_layer(input)
        bow_emb = fluid.layers.reduce_sum(emb, dim=1)
        return bow_emb


class Pair_BOWModel(Model):
    def __init__(self, conf_dict):
        super(Pair_BOWModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]
        self.padding_idx = None

        self.emb_layer = BOWEncoder(self.dict_size, self.bow_dim, self.emb_dim,
                                    self.padding_idx)
        self.bow_layer = Linear(
            input_dim=self.bow_dim, output_dim=self.bow_dim)

    def forward(self, left, pos_right, neg_right):
        bow_left = self.emb_layer(left)
        pos_bow_right = self.emb_layer(pos_right)
        neg_bow_right = self.emb_layer(neg_right)
        left_soft = fluid.layers.softsign(bow_left)
        pos_right_soft = fluid.layers.softsign(pos_bow_right)
        neg_right_soft = fluid.layers.softsign(neg_bow_right)

        left_bow = self.bow_layer(left_soft)
        pos_right_bow = self.bow_layer(pos_right_soft)
        neg_right_bow = self.bow_layer(neg_right_soft)
        pos_pred = fluid.layers.cos_sim(left_bow, pos_right_bow)
        neg_pred = fluid.layers.cos_sim(left_bow, neg_right_bow)
        return pos_pred, neg_pred


class Point_BOWModel(Model):
    def __init__(self, conf_dict):
        super(Point_BOWModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]
        self.padding_idx = None

        self.emb_layer = BOWEncoder(self.dict_size, self.bow_dim, self.emb_dim,
                                    self.padding_idx)
        self.bow_layer_po = Linear(
            input_dim=self.bow_dim * 2, output_dim=self.bow_dim)
        self.softmax_layer = Linear(
            input_dim=self.bow_dim, output_dim=2, act='softmax')

    def forward(self, left, right):
        bow_left = self.emb_layer(left)
        bow_right = self.emb_layer(right)
        left_soft = fluid.layers.softsign(bow_left)
        right_soft = fluid.layers.softsign(bow_right)

        concat = fluid.layers.concat([left_soft, right_soft], axis=1)
        concat_fc = self.bow_layer_po(concat)
        pred = self.softmax_layer(concat_fc)
        return pred
