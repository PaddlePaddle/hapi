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

import unittest
import time
import random

from callbacks import config_callbacks


class TestCallbacks(unittest.TestCase):
    def test_callback(self):
        epochs = 2
        steps = 50
        freq = 1
        eval_steps = 20
        cbks = config_callbacks(
            batch_size=128,
            epochs=epochs,
            steps=steps,
            verbose=2,
            metrics=['loss', 'acc'], )
        cbks.on_train_begin()

        logs = {'loss': 50.341673, 'acc': 0.00256}
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            for step in range(steps):
                cbks.on_train_batch_begin(step, logs)
                logs['loss'] -= random.random() * 0.1
                logs['acc'] += random.random() * 0.1
                time.sleep(0.005)
                cbks.on_train_batch_end(step, logs)
            cbks.on_epoch_end(epoch, logs)

            eval_logs = {'eval_loss': 20.341673, 'eval_acc': 0.256}
            params = {
                'eval_steps': eval_steps,
                'eval_metrics': ['eval_loss', 'eval_acc'],
                'log_freq': 10,
            }
            cbks.on_eval_begin(params)
            for step in range(eval_steps):
                cbks.on_eval_batch_begin(step, logs)
                eval_logs['eval_loss'] -= random.random() * 0.1
                eval_logs['eval_acc'] += random.random() * 0.1
                eval_logs['batch_size'] = 2
                time.sleep(0.005)
                cbks.on_eval_batch_end(step, eval_logs)
            cbks.on_eval_end(eval_logs)

        cbks.on_train_end()


if __name__ == '__main__':
    unittest.main()
