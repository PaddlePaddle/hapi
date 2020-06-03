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
hinge loss
"""

import sys
sys.path.append("../")
import paddle.fluid as fluid
from paddle.incubate.hapi.loss import Loss


class HingeLoss(Loss):
    def __init__(self, conf_dict):
        super(HingeLoss, self).__init__()
        self.margin = conf_dict["loss"]["margin"]

    def forward(self, outputs, labels=None):
        pos, neg = outputs
        loss = fluid.layers.fill_constant_batch_size_like(neg, neg.shape,
                                                          "float32", 0.0)
        loss_margin = fluid.layers.fill_constant_batch_size_like(
            neg, neg.shape, "float32", self.margin)
        sub = fluid.layers.elementwise_sub(neg, pos)
        add = fluid.layers.elementwise_add(sub, loss_margin)
        max = fluid.layers.elementwise_max(loss, add)
        loss_last = fluid.layers.reduce_mean(max)
        return loss_last
