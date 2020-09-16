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
SequenceTagging eval structure
"""

from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.utils import flatten
from paddle.static import InputSpec as Input

from sequence_tagging import SeqTagging, LacLoss, ChunkEval
from reader import LacDataset, LacDataLoader
from utils.check import check_gpu, check_version
from utils.configure import PDConfig


def main(args):
    place = paddle.set_device(args.device)
    fluid.enable_dygraph(place) if args.dynamic else None

    inputs = [
        Input(
            [None, None], 'int64', name='words'), Input(
                [None], 'int64', name='length'), Input(
                    [None, None], 'int64', name='target')
    ]
    labels = [Input([None, None], 'int64', name='labels')]

    dataset = LacDataset(args)
    eval_dataset = LacDataLoader(args, place, phase="test")

    vocab_size = dataset.vocab_size
    num_labels = dataset.num_labels
    model = paddle.Model(
        SeqTagging(
            args, vocab_size, num_labels, mode="test"),
        inputs=inputs,
        labels=labels)

    model.mode = "test"
    model.prepare(metrics=ChunkEval(num_labels))
    model.load(args.init_from_checkpoint, skip_mismatch=True)

    eval_result = model.evaluate(
        eval_dataset.dataloader, batch_size=args.batch_size)
    print("precison: %.5f" % (eval_result["precision"][0]))
    print("recall: %.5f" % (eval_result["recall"][0]))
    print("F1: %.5f" % (eval_result["F1"][0]))


if __name__ == '__main__':
    args = PDConfig(yaml_file="sequence_tagging.yaml")
    args.build()
    args.Print()

    use_gpu = True if args.device == "gpu" else False
    check_gpu(use_gpu)
    # TODO: add check for 2.0.0-alpha0 if fluid.require_version support
    # check_version()
    main(args)
