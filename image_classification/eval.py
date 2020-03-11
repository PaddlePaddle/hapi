from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('../')

import argparse
import contextlib
import time

import numpy as np

import paddle.fluid as fluid

from model import CrossEntropy
from reader import ImageNetReader
from nets import ResNet
from distributed import prepare_context, all_gather, Env, get_local_rank, get_nranks
from utils import accuracy, AverageMeter

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

    for idx, batch in enumerate(loader()):

        outputs, losses = getattr(model, mode)(
            batch[0], batch[1], device='gpu', device_ids=device_ids)

        if nranks > 1:
            outputs[0] = all_gather(outputs[0])
            label = all_gather(batch[1])
            num_samples = outputs[0].shape[0]
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
        # if idx > 5:
        #     break
    if mode == 'eval' and local_rank == 0:
        print(("[EVAL END] loss: {:0.3f} top1: {:0.5f}% top5: {:0.5f}% "
               "time: {:0.3f}, total size: {}").format(
                    total_loss.avg, total_acc1.avg,
                    total_acc5.avg, total_time / max(1, (idx - 1)), total_acc1.count))

def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if int(os.environ.get('PADDLE_TRAINERS_NUM', 1)) > 1 else fluid.CUDAPlace(0)
    guard = fluid.dygraph.guard(place) if FLAGS.dynamic else null_guard()
    if fluid.dygraph.parallel.Env().nranks > 1:
        prepare_context(place)
    imagenet_reader = ImageNetReader(0)

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
        # optim = make_optimizer(parameter_list=model.parameters())
        model.prepare(None, CrossEntropy())
        # model.save('resnet_checkpoints/{:03d}'.format(000))
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        run(model, val_loader, mode='eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=90, type=int, help="number of epoch")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r", "--resume", default=None, type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()