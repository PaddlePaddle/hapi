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
gru class
"""
import numpy as np
from paddle.fluid.dygraph import Layer, to_variable, Embedding, Linear, GRUUnit
import paddle.fluid as fluid

from paddle.incubate.hapi.model import Model
from paddle.incubate.hapi.text.text import RNN, BasicGRUCell


class GRUEncoder(Layer):
    def __init__(self, dict_size, emb_dim, gru_dim, hidden_dim, padding_idx):
        super(GRUEncoder, self).__init__()
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.gru_dim = gru_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        self.emb_layer = Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=fluid.ParamAttr(
                name='emb', initializer=fluid.initializer.Xavier()))
        cell = BasicGRUCell(
            input_size=self.gru_dim * 3, hidden_size=self.hidden_dim)
        self.gru_layer = RNN(cell=cell)
        self.proj_layer = Linear(
            input_dim=self.hidden_dim, output_dim=self.gru_dim * 3)

    def forward(self, input):
        emb = self.emb_layer(input)
        emb_proj = self.proj_layer(emb)
        gru, _ = self.gru_layer(emb_proj)
        gru = fluid.layers.reduce_max(gru, dim=1)
        gru = fluid.layers.tanh(gru)
        return gru


class Pair_GRUModel(Model):
    def __init__(self, conf_dict):
        super(Pair_GRUModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.gru_dim = conf_dict["net"]["gru_dim"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.padding_idx = None
        self.emb_layer = GRUEncoder(self.dict_size, self.emb_dim, self.gru_dim,
                                    self.hidden_dim, self.padding_idx)
        self.fc_layer = Linear(
            input_dim=self.hidden_dim, output_dim=self.hidden_dim)

    def forward(self, left, pos_right, neg_right):
        left_emb = self.emb_layer(left)
        pos_right_emb = self.emb_layer(pos_right)
        neg_right_emb = self.emb_layer(neg_right)
        left_fc = self.fc_layer(left_emb)
        pos_right_fc = self.fc_layer(pos_right_emb)
        neg_right_fc = self.fc_layer(neg_right_emb)
        pos_pred = fluid.layers.cos_sim(left_fc, pos_right_fc)
        neg_pred = fluid.layers.cos_sim(left_fc, neg_right_fc)
        return pos_pred, neg_pred


class Point_GRUModel(Model):
    def __init__(self, conf_dict):
        super(Point_GRUModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.gru_dim = conf_dict["net"]["gru_dim"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.padding_idx = None
        self.emb_layer = GRUEncoder(self.dict_size, self.emb_dim, self.gru_dim,
                                    self.hidden_dim, self.padding_idx)
        self.fc_layer_fo = Linear(
            input_dim=self.hidden_dim * 2, output_dim=self.hidden_dim)
        self.softmax_layer = Linear(
            input_dim=self.hidden_dim, output_dim=2, act='softmax')

    def forward(self, left, right):
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        concat = fluid.layers.concat([left_emb, right_emb], axis=1)
        concat_fc = self.fc_layer_fo(concat)
        pred = self.softmax_layer(concat_fc)
        return pred
