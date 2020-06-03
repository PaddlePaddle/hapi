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
cnn class
"""
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Layer, Conv2D, Pool2D, Embedding
from paddle.incubate.hapi.model import Model
from paddle.incubate.hapi.text.text import CNNEncoder


class Pair_CNNModel(Model):
    def __init__(self, conf_dict):
        super(Pair_CNNModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.padding_idx = None

        self.emb_layer = Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=fluid.ParamAttr(
                name='emb', initializer=fluid.initializer.Xavier()))
        self.encoder_layer = CNNEncoder(
            num_channels=1,
            num_filters=self.num_filters,
            filter_size=self.filter_size,
            pool_size=1,
            layer_num=1,
            act='relu')
        self.fc_layer = Linear(
            input_dim=self.num_filters, output_dim=self.hidden_dim)

    def forward(self, left, pos_right, neg_right):
        left = self.emb_layer(left)
        pos_right = self.emb_layer(pos_right)
        neg_right = self.emb_layer(neg_right)
        left_cnn = self.encoder_layer(left)
        left_cnn = fluid.layers.transpose(left_cnn, perm=[0, 2, 1])
        pos_right_cnn = self.encoder_layer(pos_right)
        pos_right_cnn = fluid.layers.transpose(pos_right_cnn, perm=[0, 2, 1])
        neg_right_cnn = self.encoder_layer(neg_right)
        neg_right_cnn = fluid.layers.transpose(neg_right_cnn, perm=[0, 2, 1])
        left_fc = self.fc_layer(left_cnn)
        pos_right_fc = self.fc_layer(pos_right_cnn)
        neg_right_fc = self.fc_layer(neg_right_cnn)
        pos_pred = fluid.layers.cos_sim(left_fc, pos_right_fc)
        neg_pred = fluid.layers.cos_sim(left_fc, neg_right_fc)
        return pos_pred, neg_pred


class Point_CNNModel(Model):
    def __init__(self, conf_dict):
        super(Point_CNNModel, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]
        self.seq_len = conf_dict["seq_len"]
        self.padding_idx = None

        self.emb_layer = Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=fluid.ParamAttr(
                name='emb', initializer=fluid.initializer.Xavier()))
        self.encoder_layer = CNNEncoder(
            num_channels=1,
            num_filters=self.num_filters,
            filter_size=self.filter_size,
            pool_size=1,
            layer_num=1,
            act='relu')

        self.fc_layer_po = Linear(
            input_dim=self.num_filters * 2, output_dim=self.hidden_dim)
        self.softmax_layer = Linear(
            input_dim=self.hidden_dim, output_dim=2, act='softmax')

    def forward(self, left, right):
        left = self.emb_layer(left)
        right = self.emb_layer(right)
        left_cnn = self.encoder_layer(left)
        left_cnn = fluid.layers.transpose(left_cnn, perm=[0, 2, 1])
        right_cnn = self.encoder_layer(right)
        right_cnn = fluid.layers.transpose(right_cnn, perm=[0, 2, 1])
        concat = fluid.layers.concat([left_cnn, right_cnn], axis=1)
        concat_fc = self.fc_layer_po(concat)
        pred = self.softmax_layer(concat_fc)
        return pred
