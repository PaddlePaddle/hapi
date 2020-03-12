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

import six
import copy

from .progressbar import ProgressBar


def config_callbacks(callbacks=None,
                     model=None,
                     batch_size=None,
                     epochs=None,
                     steps=None,
                     verbose=2,
                     log_freq=20,
                     metrics=None,
                     mode='train'):
    cbks = callbacks or []
    if not any(isinstance(k, ProgBarLogger) for k in cbks) and verbose:
        cbks = (callbacks or []) + [ProgBarLogger(log_freq)]

    cbk_list = CallbackList(cbks)
    cbk_list.set_model(model)
    metrics = metrics or [] if mode != 'test' else []
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'verbose': verbose,
        'metrics': metrics,
    }
    cbk_list.set_params(params)
    return cbk_list


class CallbackList(object):
    def __init__(self, callbacks=None):
        # copy
        self.callbacks = [c for c in callbacks]
        self.params = {}
        self.model = None

    def append(self, callback):
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def set_params(self, params):
        for c in self.callbacks:
            c.set_params(params)

    def set_model(self, model):
        for c in self.callbacks:
            c.set_model(model)

    def _call(self, name, *args):
        for c in self.callbacks:
            func = getattr(c, name)
            func(*args)

    def on_train_begin(self, logs=None):
        self._call('on_train_begin', logs)

    def on_train_end(self, logs=None):
        self._call('on_train_end', logs)

    def on_eval_begin(self, logs=None):
        self._call('on_eval_begin', logs)

    def on_eval_end(self, logs=None):
        self._call('on_eval_end', logs)

    def on_test_begin(self, logs=None):
        self._call('on_test_begin', logs)

    def on_test_end(self, logs=None):
        self._call('on_test_end', logs)

    def on_epoch_begin(self, epoch=None, logs=None):
        self._call('on_epoch_begin', epoch, logs)

    def on_epoch_end(self, epoch=None, logs=None):
        self._call('on_epoch_end', epoch, logs)

    def on_train_batch_begin(self, step=None, logs=None):
        self._call('on_train_batch_begin', step, logs)

    def on_train_batch_end(self, step=None, logs=None):
        self._call('on_train_batch_end', step, logs)

    def on_eval_batch_begin(self, step=None, logs=None):
        self._call('on_eval_batch_begin', step, logs)

    def on_eval_batch_end(self, step=None, logs=None):
        self._call('on_eval_batch_end', step, logs)

    def on_test_batch_begin(self, step=None, logs=None):
        self._call('on_test_batch_begin', step, logs)

    def on_test_batch_end(self, step=None, logs=None):
        self._call('on_test_batch_end', step, logs)


class Callback(object):
    def __init__(self):
        self.model = None
        self.params = {}

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        """
        """

    def on_train_end(self, logs=None):
        """
        """

    def on_eval_begin(self, logs=None):
        """
        """

    def on_eval_end(self, logs=None):
        """
        """

    def on_test_begin(self, logs=None):
        """
        """

    def on_test_end(self, logs=None):
        """
        """

    def on_epoch_begin(self, epoch, logs=None):
        """
        """

    def on_epoch_end(self, epoch, logs=None):
        """
        """

    def on_train_batch_begin(self, step, logs=None):
        """
        """

    def on_train_batch_end(self, step, logs=None):
        """
        """

    def on_eval_batch_begin(self, step, logs=None):
        """
        """

    def on_eval_batch_end(self, step, logs=None):
        """
        """

    def on_eval_batch_begin(self, step, logs=None):
        """
        """

    def on_eval_batch_end(self, step, logs=None):
        """
        """


class ProgBarLogger(Callback):
    def __init__(self, log_freq=1, verbose=2):
        self.epochs = None
        self.steps = None
        self.progbar = None
        self.verbose = verbose
        self.log_freq = log_freq

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics']
        assert self.train_metrics

    def on_epoch_begin(self, epoch=None, logs=None):
        self.steps = self.params['steps']
        self.verbose = self.params['verbose']
        self.train_step = 0
        if self.verbose and self.epochs:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
        self.train_progbar = ProgressBar(num=self.steps, verbose=self.verbose)

    def _updates(self, logs, mode):
        values = []
        metrics = getattr(self, '%s_metrics' % (mode))
        progbar = getattr(self, '%s_progbar' % (mode))
        steps = getattr(self, '%s_step' % (mode))
        for k in metrics:
            if k in logs:
                values.append((k, logs[k]))
        progbar.update(steps, values)

    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        steps = logs.get('steps', 1)
        self.train_step += steps
        flag = self.steps and self.train_step < self.steps

        if self.train_step % self.log_freq == 0 and self.verbose:
            # if steps is not None, last step will update in on_epoch_end
            if self.steps and self.train_step < self.steps:
                self._updates(logs, 'train')
            else:
                self._updates(logs, 'train')

    def on_epoch_end(self, step, logs=None):
        logs = logs or {}
        if self.verbose:
            self._updates(logs, 'train')

    def on_eval_begin(self, logs=None):
        self.eval_steps = logs.get('eval_steps', None)
        self.eval_metrics = logs.get('eval_metrics', [])
        self.eval_log_freq = logs.get('log_freq', 1)
        self.eval_step = 0
        self.evaled_samples = 0
        self.eval_progbar = ProgressBar(
            num=self.eval_steps, verbose=self.verbose)
        print('Eval begin...')

    def on_eval_batch_end(self, step, logs=None):
        logs = logs or {}
        steps = logs.get('steps', 1)
        self.eval_step += steps

        samples = logs.get('batch_size', 1)
        self.evaled_samples += samples
        if self.verbose and self.eval_step % self.eval_log_freq == 0:
            if self.eval_steps and self.eval_step < self.eval_steps:
                self._updates(logs, 'eval')
            else:
                self._updates(logs, 'eval')

    def on_eval_end(self, logs=None):
        logs = logs or {}
        if self.verbose:
            self._updates(logs, 'eval')
            print('Eval samples %d' % (self.evaled_samples))
