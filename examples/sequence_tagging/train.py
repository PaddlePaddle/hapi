#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
SequenceTagging network structure
"""

from __future__ import division
from __future__ import print_function

import io
import os
import sys
import math
import argparse
import numpy as np

work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(work_dir, "../"))

from hapi.model import Input, set_device
from hapi.text.sequence_tagging import SeqTagging, LacLoss, ChunkEval
from hapi.text.sequence_tagging import LacDataset, LacDataLoader
from hapi.text.sequence_tagging import check_gpu, check_version
from hapi.text.sequence_tagging import PDConfig

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [
        Input(
            [None, None], 'int64', name='words'), Input(
                [None], 'int64', name='length'), Input(
                    [None, None], 'int64', name='target')
    ]

    labels = [Input([None, None], 'int64', name='labels')]

    feed_list = None if args.dynamic else [
        x.forward() for x in inputs + labels
    ]

    dataset = LacDataset(args)
    train_dataset = LacDataLoader(args, place, phase="train")

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = SeqTagging(args, vocab_size, num_labels, mode="train")

    optim = AdamOptimizer(
        learning_rate=args.base_learning_rate,
        parameter_list=model.parameters())

    model.prepare(
        optim,
        LacLoss(),
        ChunkEval(num_labels),
        inputs=inputs,
        labels=labels,
        device=args.device)

    if args.init_from_checkpoint:
        model.load(args.init_from_checkpoint)

    if args.init_from_pretrain_model:
        model.load(args.init_from_pretrain_model, reset_optimizer=True)

    model.fit(train_dataset.dataloader,
              epochs=args.epoch,
              batch_size=args.batch_size,
              eval_freq=args.eval_freq,
              save_freq=args.save_freq,
              save_dir=args.save_dir)


if __name__ == '__main__':
    args = PDConfig(yaml_file="sequence_tagging.yaml")
    args.build()
    args.Print()

    use_gpu = True if args.device == "gpu" else False
    check_gpu(use_gpu)
    check_version()

    main(args)
