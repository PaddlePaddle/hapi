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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import argparse
import contextlib
import time

import paddle
from paddle.static import InputSpec as Input

from check import check_gpu
from cyclegan import Generator, Discriminator, GeneratorCombine, GLoss, DLoss
import data as data

step_per_epoch = 2974


def opt(parameters):
    lr_base = 0.0002
    bounds = [100, 120, 140, 160, 180]
    lr = [1., 0.8, 0.6, 0.4, 0.2, 0.1]
    bounds = [i * step_per_epoch for i in bounds]
    lr = [i * lr_base for i in lr]
    optimizer = paddle.optimizer.Adam(
        learning_rate=paddle.optimizer.lr_scheduler.PiecewiseLR(
            boundaries=bounds, values=lr),
        parameters=parameters,
        beta1=0.5)
    return optimizer


def main():
    place = paddle.set_device(FLAGS.device)
    paddle.disable_static(place) if FLAGS.dynamic else None

    im_shape = [None, 3, 256, 256]
    input_A = Input(im_shape, 'float32', 'input_A')
    input_B = Input(im_shape, 'float32', 'input_B')
    fake_A = Input(im_shape, 'float32', 'fake_A')
    fake_B = Input(im_shape, 'float32', 'fake_B')

    # Generators
    g_AB = Generator()
    g_BA = Generator()
    d_A = Discriminator()
    d_B = Discriminator()

    g = paddle.Model(
        GeneratorCombine(g_AB, g_BA, d_A, d_B), inputs=[input_A, input_B])
    g_AB = paddle.Model(g_AB, [input_A])
    g_BA = paddle.Model(g_BA, [input_B])

    # Discriminators
    d_A = paddle.Model(d_A, [input_B, fake_B])
    d_B = paddle.Model(d_B, [input_A, fake_A])

    da_params = d_A.parameters()
    db_params = d_B.parameters()
    g_params = g_AB.parameters() + g_BA.parameters()

    da_optimizer = opt(da_params)
    db_optimizer = opt(db_params)
    g_optimizer = opt(g_params)

    g_AB.prepare()
    g_BA.prepare()

    g.prepare(g_optimizer, GLoss())
    d_A.prepare(da_optimizer, DLoss())
    d_B.prepare(db_optimizer, DLoss())

    if FLAGS.resume:
        g.load(FLAGS.resume)

    loader_A = paddle.io.DataLoader(
        data.DataA(),
        places=place,
        shuffle=True,
        return_list=True,
        batch_size=FLAGS.batch_size)
    loader_B = paddle.io.DataLoader(
        data.DataB(),
        places=place,
        shuffle=True,
        return_list=True,
        batch_size=FLAGS.batch_size)

    A_pool = data.ImagePool()
    B_pool = data.ImagePool()

    for epoch in range(FLAGS.epoch):
        for i, (data_A, data_B) in enumerate(zip(loader_A, loader_B)):
            data_A = data_A[0][0] if not FLAGS.dynamic else data_A[0]
            data_B = data_B[0][0] if not FLAGS.dynamic else data_B[0]
            start = time.time()

            fake_B = g_AB.test_batch(data_A)[0]
            fake_A = g_BA.test_batch(data_B)[0]
            g_loss = g.train_batch([data_A, data_B])[0]
            fake_pb = B_pool.get(fake_B)
            da_loss = d_A.train_batch([data_B, fake_pb])[0]

            fake_pa = A_pool.get(fake_A)
            db_loss = d_B.train_batch([data_A, fake_pa])[0]

            t = time.time() - start
            if i % 20 == 0:
                print("epoch: {} | step: {:3d} | g_loss: {:.4f} | " \
                      "da_loss: {:.4f} | db_loss: {:.4f} | s/step {:.4f}".
                      format(epoch, i, g_loss[0], da_loss[0], db_loss[0], t))
        g.save('{}/{}'.format(FLAGS.checkpoint_path, epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CycleGAN Training on Cityscapes")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="Enable dygraph mode")
    parser.add_argument(
        "-p",
        "--device",
        type=str,
        default='gpu',
        help="device to use, gpu or cpu")
    parser.add_argument(
        "-e", "--epoch", default=200, type=int, help="Epoch number")
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "-o",
        "--checkpoint_path",
        type=str,
        default='checkpoint',
        help="path to save checkpoint")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    print(FLAGS)
    check_gpu(str.lower(FLAGS.device) == 'gpu')
    main()
