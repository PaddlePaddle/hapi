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

from __future__ import division
from __future__ import print_function

import argparse

from paddle import fluid
from paddle.fluid.optimizer import Momentum
from hapi.datasets.mnist import MNIST as MnistDataset

from hapi.model import Input, set_device
from hapi.loss import CrossEntropy
from hapi.metrics import Accuracy
from hapi.vision.models import LeNet


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None

    train_dataset = MnistDataset(mode='train')
    val_dataset = MnistDataset(mode='test')

    inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    model = LeNet()
    optim = Momentum(
        learning_rate=FLAGS.lr, momentum=.9, parameter_list=model.parameters())

    model.prepare(
        optim,
        CrossEntropy(),
        Accuracy(topk=(1, 2)),
        inputs,
        labels,
        device=FLAGS.device)

    if FLAGS.resume is not None:
        model.load(FLAGS.resume)

    if FLAGS.eval_only:
        model.evaluate(val_dataset, batch_size=FLAGS.batch_size)
        return

    model.fit(train_dataset,
              val_dataset,
              epochs=FLAGS.epoch,
              batch_size=FLAGS.batch_size,
              save_dir=FLAGS.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN training on MNIST")
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=10, type=int, help="number of epoch")
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
        "--output-dir",
        type=str,
        default='mnist_checkpoint',
        help="checkpoint save dir")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    parser.add_argument(
        "--eval-only", action='store_true', help="only evaluate the model")
    FLAGS = parser.parse_args()
    main()
