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
from __future__ import print_function

import argparse
import functools

import paddle

from paddle.static import InputSpec as Input
from paddle.vision.transforms import BatchCompose

from utility import add_arguments, print_arguments
from utility import SeqAccuracy, LoggerCallBack, SeqBeamAccuracy
from utility import postprocess
from seq2seq_attn import Seq2SeqAttModel, Seq2SeqAttInferModel, WeightCrossEntropy
import data

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,                 "Minibatch size.")
add_arg('test_images',       str,   None,               "The directory of images to be used for test.")
add_arg('test_list',         str,   None,               "The list file of images to be used for training.")
add_arg('init_model',        str,   'checkpoint/final', "The init model file of directory.")
add_arg('use_gpu',           bool,  True,               "Whether use GPU to train.")
add_arg('encoder_size',      int,   200,                "Encoder size.")
add_arg('decoder_size',      int,   128,                "Decoder size.")
add_arg('embedding_dim',     int,   128,                "Word vector dim.")
add_arg('num_classes',       int,   95,                 "Number classes.")
add_arg('beam_size',         int,   0,                  "If set beam size, will use beam search.")
add_arg('static',            bool,  False,              "Whether to use dygraph.")
# yapf: enable


def main(FLAGS):
    paddle.enable_static() if FLAGS.static else None
    device = paddle.set_device("gpu" if FLAGS.use_gpu else "cpu")

    # yapf: disable
    inputs = [
        Input([None, 1, 48, 384], "float32", name="pixel"),
        Input([None, None], "int64", name="label_in")
    ]
    labels = [
        Input([None, None], "int64", name="label_out"),
        Input([None, None], "float32", name="mask")
    ]
    # yapf: enable
    model = paddle.Model(
        Seq2SeqAttModel(
            encoder_size=FLAGS.encoder_size,
            decoder_size=FLAGS.decoder_size,
            emb_dim=FLAGS.embedding_dim,
            num_classes=FLAGS.num_classes),
        inputs=inputs,
        labels=labels)

    model.prepare(loss=WeightCrossEntropy(), metrics=SeqAccuracy())
    model.load(FLAGS.init_model)

    test_dataset = data.test()
    test_collate_fn = BatchCompose(
        [data.Resize(), data.Normalize(), data.PadTarget()])
    test_sampler = data.BatchSampler(
        test_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=False,
        shuffle=False)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        places=device,
        num_workers=0,
        return_list=True,
        collate_fn=test_collate_fn)

    model.evaluate(
        eval_data=test_loader,
        callbacks=[LoggerCallBack(10, 2, FLAGS.batch_size)])


def beam_search(FLAGS):
    paddle.enable_static() if FLAGS.static else None
    device = paddle.set_device("gpu" if FLAGS.use_gpu else "cpu")

    # yapf: disable
    inputs = [
        Input([None, 1, 48, 384], "float32", name="pixel"),
        Input([None, None], "int64", name="label_in")
    ]
    labels = [
        Input([None, None], "int64", name="label_out"),
        Input([None, None], "float32", name="mask")
    ]
    # yapf: enable

    model = paddle.Model(
        Seq2SeqAttInferModel(
            encoder_size=FLAGS.encoder_size,
            decoder_size=FLAGS.decoder_size,
            emb_dim=FLAGS.embedding_dim,
            num_classes=FLAGS.num_classes,
            beam_size=FLAGS.beam_size),
        inputs=inputs,
        labels=labels)

    model.prepare(metrics=SeqBeamAccuracy())
    model.load(FLAGS.init_model)

    test_dataset = data.test()
    test_collate_fn = BatchCompose(
        [data.Resize(), data.Normalize(), data.PadTarget()])
    test_sampler = data.BatchSampler(
        test_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=False,
        shuffle=False)
    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        places=device,
        num_workers=0,
        return_list=True,
        collate_fn=test_collate_fn)

    model.evaluate(
        eval_data=test_loader,
        callbacks=[LoggerCallBack(10, 2, FLAGS.batch_size)])


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    if FLAGS.beam_size:
        beam_search(FLAGS)
    else:
        main(FLAGS)
