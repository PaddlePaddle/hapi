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

import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.incubate.hapi.model import Input, set_device

from sequence_tagging import SeqTagging, LacLoss, ChunkEval
from reader import LacDataset, LacDataLoader
from utils.check import check_gpu, check_version
from utils.configure import PDConfig


def main(args):
    place = set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [
        Input(
            [None, None], 'int64', name='words'),
        Input(
            [None], 'int64', name='length'),
        Input(
            [None, None], 'int64', name='target'),
    ]

    labels = [Input([None, None], 'int64', name='labels')]

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
    # TODO: add check for 2.0.0-alpha0 if fluid.require_version support
    # check_version()

    main(args)
