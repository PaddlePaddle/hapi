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
log loss
"""

import sys
sys.path.append("../")
import paddle.fluid as fluid
from paddle.incubate.hapi.model import Loss


class LogLoss(Loss):
    def __init__(self, conf_dict):
        super(LogLoss, self).__init__()

    def forward(self, outputs, labels=None):
        pos, neg = outputs
        loss = fluid.layers.sigmoid(neg - pos)
        avg_loss = fluid.layers.reduce_mean(loss)
        return loss
