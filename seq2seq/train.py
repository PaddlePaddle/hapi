# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import random
from args import parse_args
from functools import partial

import numpy as np
import paddle
from paddle.static import InputSpec as Input

from seq2seq import Seq2Seq, CrossEntropyCriterion
from reader import create_data_loader
from utility import PPL, TrainCallback


def do_train(args):
    device = paddle.set_device("gpu" if args.use_gpu else "cpu")
    paddle.enable_static() if not args.eager_run else None

    if args.enable_ce:
        paddle.manual_seed(102)

    # define model
    inputs = [
        Input(
            [None, None], "int64", name="src_word"),
        Input(
            [None], "int64", name="src_length"),
        Input(
            [None, None], "int64", name="trg_word"),
    ]
    labels = [
        Input(
            [None], "int64", name="trg_length"),
        Input(
            [None, None, 1], "int64", name="label"),
    ]

    # def dataloader
    [train_loader, eval_loader], pad_id = create_data_loader(args, device)

    model = paddle.Model(
        Seq2Seq(args.src_vocab_size, args.tar_vocab_size, args.hidden_size,
                args.hidden_size, args.num_layers, args.attention,
                args.dropout, pad_id),
        inputs=inputs,
        labels=labels)
    grad_clip = paddle.nn.GradientClipByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        grad_clip=grad_clip)

    ppl_metric = PPL(reset_freq=100)  # ppl for every 100 batches
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)
    model.fit(train_data=train_loader,
              eval_data=eval_loader,
              epochs=args.max_epoch,
              eval_freq=1,
              save_freq=1,
              save_dir=args.model_path,
              callbacks=[TrainCallback(ppl_metric, args.log_freq)])


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
