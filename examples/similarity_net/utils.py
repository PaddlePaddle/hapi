# -*- encoding:utf-8 -*-
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
"""
SimNet utilities.
"""
import argparse
import time
import sys
import re
import os
import six
import numpy as np
import paddle.fluid as fluid
import io
import pickle
import warnings
from functools import partial
from hapi.configure import ArgumentGroup, str2bool
"""
******functions for file processing******
"""


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    f = io.open(file_path, "r", encoding="utf8")
    for line in f:
        items = line.strip("\n").split("\t")
        if items[0] not in vocab:
            vocab[items[0]] = int(items[1])
    vocab["<unk>"] = 0
    return vocab


def get_result_file(args):
    """
    Get Result File
    Args:
      conf_dict: Input path config
      samples_file_path: Data path of real training
      predictions_file_path: Prediction results path
    Returns:
      result_file: merge sample and predict result

    """
    with io.open(args.test_data_dir, "r", encoding="utf8") as test_file:
        with io.open(
                "predictions.txt", "r", encoding="utf8") as predictions_file:
            with io.open(
                    args.test_result_path, "w",
                    encoding="utf8") as test_result_file:
                test_datas = [line.strip("\n") for line in test_file]
                predictions = [line.strip("\n") for line in predictions_file]
                for test_data, prediction in zip(test_datas, predictions):
                    test_result_file.write(test_data + "\t" + prediction +
                                           "\n")
    os.remove("predictions.txt")


def import_class(module_path, module_name, class_name):
    """
    Load class dynamically
    Args:
      module_path: The current path of the module
      module_name: The module name
      class_name: The name of class in the import module
    Return:
      Return the attribute value of the class object
    """
    if module_path:
        sys.path.append(module_path)
    module = __import__(module_name)
    return getattr(module, class_name)


"""
******functions for string processing******
"""


def pattern_match(pattern, line):
    """
    Check whether a string is matched
    Args:
      pattern: mathing pattern
      line : input string
    Returns:
      True/False
    """
    if re.match(pattern, line):
        return True
    else:
        return False


"""
******functions for parameter processing******
"""


def print_progress(task_name, percentage, style=0):
    """
    Print progress bar
    Args:
      task_name: The name of the current task
      percentage: Current progress
      style: Progress bar form
    """
    styles = ['#', '█']
    mark = styles[style] * percentage
    mark += ' ' * (100 - percentage)
    status = '%d%%' % percentage if percentage < 100 else 'Finished'
    sys.stdout.write('%+20s [%s] %s\r' % (task_name, mark, status))
    sys.stdout.flush()
    time.sleep(0.002)


class ArgConfig(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        model_g = ArgumentGroup(parser, "model",
                                "model configuration and paths.")
        model_g.add_arg("config_path", str, None,
                        "Path to the json file for EmoTect model config.")
        model_g.add_arg("init_checkpoint", str, None,
                        "Init checkpoint to resume training from.")
        model_g.add_arg("output_dir", str, None,
                        "Directory path to save checkpoints")
        model_g.add_arg("task_mode", str, None,
                        "task mode: pairwise or pointwise")

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
        train_g.add_arg("save_steps", int, 20,
                        "The steps interval to save checkpoints.")
        train_g.add_arg("validation_steps", int, 100,
                        "The steps interval to evaluate model performance.")

        infer_g = ArgumentGroup(parser, "inferring", "inferring related")
        infer_g.add_arg("test_result_path", str, "test_result",
                        "Directory path to test result.")
        infer_g.add_arg("infer_result_path", str, "infer_result.txt",
                        "Directory path to infer result.")

        data_g = ArgumentGroup(
            parser, "data",
            "Data paths, vocab paths and data processing options")
        data_g.add_arg("train_data_dir", str, None,
                       "Directory path to training data.")
        data_g.add_arg("valid_data_dir", str, None,
                       "Directory path to valid data.")
        data_g.add_arg("test_data_dir", str, None,
                       "Directory path to testing data.")
        data_g.add_arg("infer_data_dir", str, None,
                       "Directory path to infer data.")
        data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
        data_g.add_arg("batch_size", int, 32,
                       "Total examples' number in batch for training.")
        data_g.add_arg("seq_len", int, 32, "The length of each sentence.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_cuda", bool, False,
                           "If set, use GPU for training.")
        run_type_g.add_arg(
            "task_name", str, None,
            "The name of task to perform sentiment classification.")
        run_type_g.add_arg("do_train", bool, False,
                           "Whether to perform training.")
        run_type_g.add_arg("do_valid", bool, False, "Whether to perform dev.")
        #run_type_g.add_arg("do_test", bool, False, "Whether to perform testing.")
        run_type_g.add_arg("do_infer", bool, False,
                           "Whether to perform inference.")
        run_type_g.add_arg("compute_accuracy", bool, False,
                           "Whether to compute accuracy.")
        run_type_g.add_arg(
            "lamda", float, 0.91,
            "When task_mode is pairwise, lamda is the threshold for calculating the accuracy."
        )

        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        self.custom_g = custom_g

        #parser.add_argument('--enable_ce',action='store_true',help='If set, run the task with continuous evaluation logs.')

        self.parser = parser

    def add_arg(self, name, dtype, default, descrip):
        self.custom_g.add_arg(name, dtype, default, descrip)

    def build_conf(self):
        return self.parser.parse_args()


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def get_softmax(preds):
    """
    compute sotfmax
    """
    _exp = np.exp(preds)
    return _exp / np.sum(_exp, axis=1, keepdims=True)


def get_sigmoid(preds):
    """
    compute sigmoid
    """
    return 1 / (1 + np.exp(-preds))


def get_accuracy(preds, labels, mode, lamda=0.958):
    """
    compute accuracy
    """
    if mode == "pairwise":
        preds = np.array(list(map(lambda x: 1 if x[1] >= lamda else 0, preds)))
    else:
        preds = np.array(list(map(lambda x: np.argmax(x), preds)))
    labels = np.squeeze(labels)
    return np.mean(preds[0:len(labels)] == labels)
