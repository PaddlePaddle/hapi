# -*- encoding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import io
import warnings
import argparse
import multiprocessing

import paddle
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader
from functools import partial, reduce
import numpy as np
import reader
import config
from utils import load_vocab, import_class, get_accuracy, ArgConfig, print_arguments

from paddle.incubate.hapi.metrics import Accuracy
from paddle.incubate.hapi.model import set_device, Model, Input, Loss, CrossEntropy


def train(conf_dict, args):
    device = set_device("cpu")
    fluid.enable_dygraph(device)

    # load auc method
    metric = fluid.metrics.Auc(name="auc")

    def valid_and_test(pred_list, process, mode):
        """
        return auc and acc
        """
        pred_list = np.vstack(pred_list)
        if mode == "test":
            label_list = process.get_test_label()
        elif mode == "valid":
            label_list = process.get_valid_label()
        if args.task_mode == "pairwise":
            pred_list = (pred_list + 1) / 2
            pred_list = np.hstack(
                (np.ones_like(pred_list) - pred_list, pred_list))
        metric.reset()
        metric.update(pred_list, label_list)
        auc = metric.eval()
        if args.compute_accuracy:
            acc = get_accuracy(pred_list, label_list, args.task_mode,
                               args.lamda)
            return auc, acc
        else:
            return auc

    # loading vocabulary
    vocab = load_vocab(args.vocab_path)
    # get vocab size
    conf_dict['dict_size'] = len(vocab)
    conf_dict['seq_len'] = args.seq_len
    # Load network structure dynamically
    model = import_class("./nets", conf_dict["net"]["module_name"],
                         conf_dict["net"]["class_name"])(conf_dict)
    loss = import_class("./nets/losses", conf_dict["loss"]["module_name"],
                        conf_dict["loss"]["class_name"])(conf_dict)
    # Load Optimization method
    learning_rate = conf_dict["optimizer"]["learning_rate"]
    optimizer_name = conf_dict["optimizer"]["class_name"]
    if optimizer_name == 'SGDOptimizer':
        optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate, parameter_list=model.parameters())
    elif optimizer_name == 'AdamOptimizer':
        beta1 = conf_dict["optimizer"]["beta1"]
        beta2 = conf_dict["optimizer"]["beta2"]
        epsilon = conf_dict["optimizer"]["epsilon"]
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameter_list=model.parameters())

    global_step = 0
    valid_step = 0
    losses = []

    # define dataloader
    simnet_process = reader.SimNetProcessor(args, vocab)
    train_pyreader = DataLoader.from_generator(
        capacity=16, return_list=True, use_double_buffer=True)
    get_train_examples = simnet_process.get_reader("train", epoch=args.epoch)
    train_pyreader.set_sample_list_generator(
        fluid.io.batch(
            get_train_examples, batch_size=args.batch_size),
        places=device)
    if args.do_valid:
        valid_pyreader = DataLoader.from_generator(
            capacity=16, return_list=True, use_double_buffer=True)
        get_valid_examples = simnet_process.get_reader("valid")
        valid_pyreader.set_sample_list_generator(
            fluid.io.batch(
                get_valid_examples, batch_size=args.batch_size),
            places=device)
        pred_list = []

    if args.task_mode == "pairwise":
        inputs = [
            Input(
                [None, 1], 'int64', name='input_left'), Input(
                    [None, 1], 'int64', name='pos_right'), Input(
                        [None, 1], 'int64', name='neg_right')
        ]

        model.prepare(
            inputs=inputs,
            optimizer=optimizer,
            loss_function=loss,
            device=device)

        for left, pos_right, neg_right in train_pyreader():
            input_left = fluid.layers.reshape(left, shape=[-1, 1])
            pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
            neg_right = fluid.layers.reshape(neg_right, shape=[-1, 1])

            final_loss = model.train_batch([input_left, pos_right, neg_right])
            print("train_steps: %d, train_loss: %f" %
                  (global_step, final_loss[0][0]))
            losses.append(np.mean(final_loss))
            global_step += 1

            if args.do_valid and global_step % args.validation_steps == 0:
                for left, pos_right, neg_right in valid_pyreader():
                    input_left = fluid.layers.reshape(left, shape=[-1, 1])
                    pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
                    neg_right = fluid.layers.reshape(neg_right, shape=[-1, 1])

                    result, _ = model.test_batch(
                        [input_left, pos_right, neg_right])
                    pred_list += list(result)
                    valid_step += 1

                valid_result = valid_and_test(pred_list, simnet_process,
                                              "valid")
                if args.compute_accuracy:
                    valid_auc, valid_acc = valid_result
                    print(
                        "valid_steps: %d, valid_auc: %f, valid_acc: %f, valid_loss: %f"
                        % (global_step, valid_auc, valid_acc, np.mean(losses)))
                else:
                    valid_auc = valid_result
                    print("valid_steps: %d, valid_auc: %f, valid_loss: %f" %
                          (global_step, valid_auc, np.mean(losses)))

            if global_step % args.save_steps == 0:
                model_save_dir = os.path.join(args.output_dir,
                                              conf_dict["model_path"])
                model_path = os.path.join(model_save_dir, str(global_step))

                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model.save(model_path)

    else:
        inputs = [
            Input(
                [None, 1], 'int64', name='left'), Input(
                    [None, 1], 'int64', name='right')
        ]
        label = [Input([None, 1], 'int64', name='neg_right')]

        model.prepare(
            inputs=inputs,
            optimizer=optimizer,
            loss_function=loss,
            device=device)

        for left, right, label in train_pyreader():
            left = fluid.layers.reshape(left, shape=[-1, 1])
            right = fluid.layers.reshape(right, shape=[-1, 1])
            label = fluid.layers.reshape(label, shape=[-1, 1])

            final_loss = model.train_batch([left, right], [label])
            print("train_steps: %d, train_loss: %f" %
                  (global_step, final_loss[0][0]))
            losses.append(np.mean(final_loss))
            global_step += 1

            if args.do_valid and global_step % args.validation_steps == 0:
                for left, right, label in valid_pyreader():
                    valid_left = fluid.layers.reshape(left, shape=[-1, 1])
                    valid_right = fluid.layers.reshape(right, shape=[-1, 1])
                    valid_label = fluid.layers.reshape(label, shape=[-1, 1])

                    result, _ = model.test_batch(
                        [valid_left, valid_right, valid_right])
                    pred_list += list(result)
                    valid_step += 1

                valid_result = valid_and_test(pred_list, simnet_process,
                                              "valid")
                if args.compute_accuracy:
                    valid_auc, valid_acc = valid_result
                    print(
                        "valid_steps: %d, valid_auc: %f, valid_acc: %f, valid_loss: %f"
                        % (global_step, valid_auc, valid_acc, np.mean(losses)))
                else:
                    valid_auc = valid_result
                    print("valid_steps: %d, valid_auc: %f, valid_loss: %f" %
                          (global_step, valid_auc, np.mean(losses)))

            if global_step % args.save_steps == 0:
                model_save_dir = os.path.join(args.output_dir,
                                              conf_dict["model_path"])
                model_path = os.path.join(model_save_dir, str(global_step))

                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model.save(model_path)


def test(conf_dict, args):
    device = set_device("cpu")
    fluid.enable_dygraph(device)

    metric = fluid.metrics.Auc(name="auc")

    def valid_and_test(pred_list, process, mode):
        """
        return auc and acc
        """
        pred_list = np.vstack(pred_list)
        if mode == "test":
            label_list = process.get_test_label()
        elif mode == "valid":
            label_list = process.get_valid_label()
        if args.task_mode == "pairwise":
            pred_list = (pred_list + 1) / 2
            pred_list = np.hstack(
                (np.ones_like(pred_list) - pred_list, pred_list))
        metric.reset()
        metric.update(pred_list, label_list)
        auc = metric.eval()
        if args.compute_accuracy:
            acc = get_accuracy(pred_list, label_list, args.task_mode,
                               args.lamda)
            return auc, acc
        else:
            return auc

    # loading vocabulary
    vocab = load_vocab(args.vocab_path)
    # get vocab size
    conf_dict['dict_size'] = len(vocab)
    conf_dict['seq_len'] = args.seq_len
    # Load network structure dynamically
    model = import_class("./nets", conf_dict["net"]["module_name"],
                         conf_dict["net"]["class_name"])(conf_dict)
    model.load(args.init_checkpoint)

    simnet_process = reader.SimNetProcessor(args, vocab)
    test_pyreader = DataLoader.from_generator(
        capacity=16, return_list=True, use_double_buffer=True)
    get_test_examples = simnet_process.get_reader("test")
    test_pyreader.set_sample_list_generator(
        fluid.io.batch(
            get_test_examples, batch_size=args.batch_size),
        places=device)

    pred_list = []
    test_step = 0

    if args.task_mode == "pairwise":
        inputs = [
            Input(
                [None, 1], 'int64', name='input_left'), Input(
                    [None, 1], 'int64', name='pos_right'), Input(
                        [None, 1], 'int64', name='pos_right')
        ]

        model.prepare(inputs=inputs, device=device)

        for left, pos_right, neg_right in test_pyreader():
            input_left = fluid.layers.reshape(left, shape=[-1, 1])
            pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
            neg_right = fluid.layers.reshape(pos_right, shape=[-1, 1])

            final_pred, _ = model.test_batch(
                [input_left, pos_right, neg_right])
            pred_list += list(final_pred)
            test_step += 1

        test_result = valid_and_test(pred_list, simnet_process, "test")
        if args.compute_accuracy:
            test_auc, test_acc = test_result
            print("test_steps: %d, test_auc: %f, test_acc: %f" %
                  (test_step, test_auc, test_acc))
        else:
            test_auc = test_result
            print("test_steps: %d, test_auc: %f" % (test_step, test_auc))

    else:
        inputs = [
            Input(
                [None, 1], 'int64', name='left'), Input(
                    [None, 1], 'int64', name='right')
        ]

        model.prepare(inputs=inputs, device=device)

        for left, right, label in test_pyreader():
            left = fluid.layers.reshape(left, shape=[-1, 1])
            right = fluid.layers.reshape(right, shape=[-1, 1])
            label = fluid.layers.reshape(label, shape=[-1, 1])

            final_pred = model.test_batch([left, right])
            print(final_pred)
            pred_list += list(final_pred)
            test_step += 1

        test_result = valid_and_test(pred_list, simnet_process, "test")
        if args.compute_accuracy:
            test_auc, test_acc = test_result
            print("test_steps: %d, test_auc: %f, test_acc: %f" %
                  (test_step, test_auc, test_acc))
        else:
            test_auc = test_result
            print("test_steps: %d, test_auc: %f" % (test_step, test_auc))


def infer(conf_dict, args):
    device = set_device("cpu")
    fluid.enable_dygraph(device)

    # loading vocabulary
    vocab = load_vocab(args.vocab_path)
    # get vocab size
    conf_dict['dict_size'] = len(vocab)
    conf_dict['seq_len'] = args.seq_len
    # Load network structure dynamically
    model = import_class("./nets", conf_dict["net"]["module_name"],
                         conf_dict["net"]["class_name"])(conf_dict)
    model.load(args.init_checkpoint)

    simnet_process = reader.SimNetProcessor(args, vocab)
    get_infer_examples = simnet_process.get_infer_reader
    infer_pyreader = DataLoader.from_generator(
        capacity=16, return_list=True, use_double_buffer=True)
    infer_pyreader.set_sample_list_generator(
        fluid.io.batch(
            get_infer_examples, batch_size=args.batch_size),
        places=device)
    pred_list = []

    if args.task_mode == "pairwise":
        inputs = [
            Input(
                [None, 1], 'int64', name='input_left'), Input(
                    [None, 1], 'int64', name='pos_right')
        ]

        model.prepare(inputs=inputs, device=device)

        for left, pos_right in infer_pyreader():
            input_left = fluid.layers.reshape(left, shape=[-1, 1])
            pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
            neg_right = fluid.layers.reshape(pos_right, shape=[-1, 1])

            final_pred, _ = model.test_batch(
                [input_left, pos_right, neg_right])
            pred_list += list(
                map(lambda item: str((item[0] + 1) / 2), final_pred))
            print(pred_list)

    else:
        inputs = [
            Input(
                [None, 1], 'int64', name='left'), Input(
                    [None, 1], 'int64', name='right')
        ]

        model.prepare(inputs=inputs, device=device)

        for left, right in infer_pyreader():
            left = fluid.layers.reshape(left, shape=[-1, 1])
            right = fluid.layers.reshape(right, shape=[-1, 1])
            # label = fluid.layers.reshape(label, shape=[-1, 1])

            final_pred = model.test_batch([left, right])
            print(final_pred)
            pred_list += list(
                map(lambda item: str((item[0] + 1) / 2), final_pred))

    with io.open(args.infer_result_path, "w", encoding="utf8") as infer_file:
        for _data, _pred in zip(simnet_process.get_infer_data(),
                                int(pred_list)):
            infer_file.write(_data + "\t" + _pred + "\n")


if __name__ == '__main__':
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)
    conf_dict = config.SimNetConfig(args)

    if args.do_train:
        train(conf_dict, args)
    elif args.do_test:
        test(conf_dict, args)
    elif args.do_infer:
        infer(conf_dict, args)
    else:
        raise ValueError("one of do_train and do_infer must be True")
