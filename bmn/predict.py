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

import argparse
import sys
import os
import logging
import paddle

from modeling import bmn, BmnLoss
from bmn_metric import BmnMetric
from reader import BmnDataset
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("BMN inference.")
    parser.add_argument(
        "-s", "--static", action='store_true', help="enable static mode")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help='gpu or cpu, default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='infer file list, None to use config file setting.')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='output dir path, None to use config file setting.')
    parser.add_argument(
        '--result_path',
        type=str,
        default=None,
        help='output dir path after post processing,  None to use config file setting.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Prediction
def infer_bmn(args):
    paddle.enable_static() if args.static else None
    device = paddle.set_device(args.device)

    #config setting
    config = parse_config(args.config_file)
    infer_cfg = merge_configs(config, 'infer', vars(args))

    feat_dim = config.MODEL.feat_dim
    tscale = config.MODEL.tscale
    dscale = config.MODEL.dscale
    prop_boundary_ratio = config.MODEL.prop_boundary_ratio
    num_sample = config.MODEL.num_sample
    num_sample_perbin = config.MODEL.num_sample_perbin

    #data
    infer_dataset = BmnDataset(infer_cfg, 'infer')

    #model
    model = bmn(tscale,
                dscale,
                feat_dim,
                prop_boundary_ratio,
                num_sample,
                num_sample_perbin,
                mode='infer',
                pretrained=args.weights is None)

    model.prepare(metrics=BmnMetric(config, mode='infer'))

    # load checkpoint
    if args.weights is not None:
        assert os.path.exists(
            args.weights), "Given weight dir {} not exist.".format(
                args.weights)
        logger.info('load test weights from {}'.format(args.weights))
        model.load(args.weights)

    # here use model.eval instead of model.test, as post process is required in our case
    model.evaluate(
        eval_data=infer_dataset,
        batch_size=infer_cfg.TEST.batch_size,
        num_workers=infer_cfg.TEST.num_workers,
        log_freq=args.log_interval)

    logger.info("[INFER] infer finished")


if __name__ == '__main__':
    args = parse_args()
    infer_bmn(args)
