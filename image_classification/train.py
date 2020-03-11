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
import os
import sys
sys.path.append('../')

import time
import math
import numpy as np
import paddle.fluid as fluid

from model import CrossEntropy
from reader import ImageNetReader
from utils import AverageMeter, accuracy
from distributed import prepare_context, all_gather, Env, get_nranks, get_local_rank
from nets import ResNet


def make_optimizer(parameter_list=None):
    total_images = 1281167
    base_lr = FLAGS.lr
    momentum = 0.9
    weight_decay = 1e-4
    step_per_epoch = int(math.floor(float(total_images) / FLAGS.batch_size))
    boundaries = [step_per_epoch * e for e in [30, 60, 90]]
    values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
    learning_rate = fluid.layers.piecewise_decay(
        boundaries=boundaries, values=values)
    learning_rate = fluid.layers.linear_lr_warmup(
        learning_rate=learning_rate,
        warmup_steps=5 * step_per_epoch,
        start_lr=0.,
        end_lr=base_lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(weight_decay),
        parameter_list=parameter_list)
    return optimizer


def run(model, loader, mode='train'):
    total_loss = AverageMeter()
    total_acc1 = AverageMeter()
    total_acc5 = AverageMeter()
    total_time = 0.0 #AverageMeter()
    start = time.time()
    nranks = get_nranks()
    local_rank = get_local_rank()
    device_ids = [0] if nranks > 1 else list(range(FLAGS.num_devices))
    start = time.time()

    # num_samples = 0
    for idx, batch in enumerate(loader()):
        outputs, losses = getattr(model, mode)(
            batch[0], batch[1], device='gpu', device_ids=device_ids)

        if nranks > 1 and mode == 'eval':
            outputs[0] = all_gather(outputs[0])
            label = all_gather(batch[1])
            num_samples = outputs[0].shape[0]
            # hard code here, 50000 is imagenet val set length
            if total_acc1.count + num_samples > 50000:
                num_samples = 50000 - total_acc1.count
                outputs[0] = outputs[0][:num_samples, ...]
                label = label[:num_samples, ...]
        else:
            label = batch[1]
            num_samples = outputs[0].shape[0]

        top1, top5 = accuracy(outputs[0], label, topk=(1, 5))

        total_loss.update(np.sum(losses), 1)
        total_acc1.update(top1, num_samples)
        total_acc5.update(top5, num_samples)

        if idx > 1:  # skip first two steps
            total_time += time.time() - start

        if idx % 10 == 0 and local_rank == 0:
            print(("{:04d} loss: {:0.3f} top1: {:0.5f}% top5: {:0.5f}% "
                   "time: {:0.3f}").format(
                       idx, total_loss.avg, total_acc1.avg,
                       total_acc5.avg, total_time / max(1, (idx - 1))))
        start = time.time()
 
    if mode == 'eval' and local_rank == 0:
        print(("[EVAL END] loss: {:0.3f} top1: {:0.5f}% top5: {:0.5f}% "
               "time: {:0.3f}, total size: {}").format(
                    total_loss.avg, total_acc1.avg,
                    total_acc5.avg, total_time / max(1, (idx - 1)), total_acc1.count))


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    epoch = FLAGS.epoch
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if fluid.dygraph.parallel.Env().nranks > 1 else fluid.CUDAPlace(0)
    guard = fluid.dygraph.guard(place) if FLAGS.dynamic else null_guard()
    if fluid.dygraph.parallel.Env().nranks > 1:
        prepare_context(place)

    imagenet_reader = ImageNetReader(0)
    train_loader = fluid.io.xmap_readers(
        lambda batch: (np.array([b[0] for b in batch]),
                       np.array([b[1] for b in batch]).reshape(-1, 1)),
        imagenet_reader.train(settings=FLAGS),
        process_num=4, buffer_size=4
        )
    val_loader = fluid.io.xmap_readers(
        lambda batch: (np.array([b[0] for b in batch]),
                       np.array([b[1] for b in batch]).reshape(-1, 1)),
        imagenet_reader.val(settings=FLAGS),
        process_num=4, buffer_size=4
        )

    if not os.path.exists('resnet_checkpoints'):
        os.mkdir('resnet_checkpoints')

    with guard:
        model = ResNet()
        optim = make_optimizer(parameter_list=model.parameters())
        model.prepare(optim, CrossEntropy())
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        for e in range(epoch):
            print("======== train epoch {} ========".format(e))
            run(model, train_loader)
            model.save('resnet_checkpoints/{:02d}'.format(e))
            print("======== eval epoch {} ========".format(e))
            run(model, val_loader, mode='eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=120, type=int, help="number of epoch")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=256, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r", "--resume", default=None, type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()
