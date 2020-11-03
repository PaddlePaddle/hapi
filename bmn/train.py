#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.distributed as dist
import argparse
import logging
import sys
import os

from reader import BmnDataset
from config_utils import *
from modeling import bmn, BmnLoss

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle high level api of BMN.")
    parser.add_argument(
        "-s", "--static", action='store_true', help="enable static mode")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None for read from config file.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='filename to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help='gpu or cpu, default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, None for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="checkpoint",
        help='path to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Optimizer
def optimizer(cfg, parameter_list):
    bd = [cfg.TRAIN.lr_decay_iter]
    base_lr = cfg.TRAIN.learning_rate
    lr_decay = cfg.TRAIN.learning_rate_decay
    l2_weight_decay = cfg.TRAIN.l2_weight_decay
    lr = [base_lr, base_lr * lr_decay]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=parameter_list,
        weight_decay=l2_weight_decay)
    return optimizer


# TRAIN
def train_bmn(args):
    paddle.enable_static() if args.static else None
    device = paddle.set_device(args.device)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    #config setting
    config = parse_config(args.config_file)
    train_cfg = merge_configs(config, 'train', vars(args))
    val_cfg = merge_configs(config, 'valid', vars(args))

    feat_dim = config.MODEL.feat_dim
    tscale = config.MODEL.tscale
    dscale = config.MODEL.dscale
    prop_boundary_ratio = config.MODEL.prop_boundary_ratio
    num_sample = config.MODEL.num_sample
    num_sample_perbin = config.MODEL.num_sample_perbin

    # data
    train_dataset = BmnDataset(train_cfg, 'train')
    val_dataset = BmnDataset(val_cfg, 'valid')

    # model
    model = bmn(tscale,
                dscale,
                feat_dim,
                prop_boundary_ratio,
                num_sample,
                num_sample_perbin,
                mode='train',
                pretrained=False)
    optim = optimizer(config, parameter_list=model.parameters())
    model.prepare(optimizer=optim, loss=BmnLoss(tscale, dscale))

    # if resume weights is given, load resume weights directly
    if args.resume is not None:
        assert os.path.exists(
            args.resume + '.pdparams'
        ), "Given weight dir {}.pdparams not exist.".format(args.resume)
        assert os.path.exists(args.resume + '.pdopt'
                              ), "Given weight dir {}.pdopt not exist.".format(
                                  args.resume)
        model.load(args.resume + '.pdparams')
        optim.load(args.resume + '.pdopt')

    model.fit(
        train_data=train_dataset,
        eval_data=val_dataset,
        batch_size=train_cfg.TRAIN.batch_size,  #batch_size of one card
        epochs=train_cfg.TRAIN.epoch,
        eval_freq=args.valid_interval,
        log_freq=args.log_interval,
        save_dir=args.save_dir,
        shuffle=train_cfg.TRAIN.use_shuffle,
        num_workers=train_cfg.TRAIN.num_workers,
        drop_last=True)


if __name__ == "__main__":
    args = parse_args()
    dist.spawn(
        train_bmn, args=(args, ),
        nprocs=4)  # if single-card training please set "nprocs=1"
