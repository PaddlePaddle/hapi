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
from __future__ import print_function

import argparse
import contextlib
import sys
import os

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.optimizer import Momentum
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

from model import Model, CrossEntropy, Input


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=None,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__('SimpleConv')

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()
        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            800,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def report_eval_res(self, eval_reader, outstream=sys.stdout):
        def accuracy(pred, label, topk=(1, )):
            maxk = max(topk)
            pred = np.argsort(pred)[:, ::-1][:, :maxk]
            correct = (pred == np.repeat(label, maxk, 1))

            batch_size = label.shape[0]
            res = []
            for k in topk:
                correct_k = correct[:, :k].sum()
                res.append((correct_k, batch_size))
            return res

        if self._runtime_states["step"] % 300 == 0 and self._runtime_states[
                "step"] > 0:
            accum_acc = []
            for epoch_idx, step_idx, batch in eval_reader:
                outputs, losses = self.eval(batch[0], batch[1])
                acc = accuracy(outputs[0], batch[1])
                accum_acc.extend(acc)

            correct = 0.
            total = 0.
            for a, b in accum_acc:
                correct += a
                total += b

            self.report("[eval] eval acc %0.3f" % (correct / total))

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self._fc(x)
        return x


def accuracy(pred, label, topk=(1, )):
    maxk = max(topk)
    pred = np.argsort(pred)[:, ::-1][:, :maxk]
    correct = (pred == np.repeat(label, maxk, 1))

    batch_size = label.shape[0]
    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(100.0 * correct_k / batch_size)
    return res


def build_reader(loader, epoch_num):
    step_idx = 0
    for epoch_idx in range(epoch_num):
        for batch in loader():
            yield epoch_idx, step_idx, batch
            step_idx += 1


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    guard = fluid.dygraph.guard() if FLAGS.dynamic else null_guard()

    if not os.path.exists('mnist_checkpoints'):
        os.mkdir('mnist_checkpoints')

    train_loader = fluid.io.xmap_readers(
        lambda b: [np.array([x[0] for x in b]).reshape(-1, 1, 28, 28),
                   np.array([x[1] for x in b]).reshape(-1, 1)],
        paddle.batch(fluid.io.shuffle(paddle.dataset.mnist.train(), 6e4),
                     batch_size=FLAGS.batch_size, drop_last=True), 1, 1)
    val_loader = fluid.io.xmap_readers(
        lambda b: [np.array([x[0] for x in b]).reshape(-1, 1, 28, 28),
                   np.array([x[1] for x in b]).reshape(-1, 1)],
        paddle.batch(paddle.dataset.mnist.test(),
                     batch_size=FLAGS.batch_size, drop_last=True), 1, 1)

    train_reader = build_reader(train_loader, FLAGS.epoch)
    val_reader = build_reader(val_loader, 1)

    with guard:
        model = MNIST()
        optim = Momentum(
            learning_rate=FLAGS.lr,
            momentum=.9,
            parameter_list=model.parameters())
        inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]
        model.prepare(optim, CrossEntropy(), inputs, labels)
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        model.fit(train_reader, val_reader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN training on MNIST")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=100, type=int, help="number of epoch")
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    main()
