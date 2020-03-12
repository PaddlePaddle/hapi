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

from __future__ import absolute_import

import inspect
import os
import pickle
from collections import OrderedDict

import numpy as np

from paddle import fluid
from paddle.fluid.framework import in_dygraph_mode, Variable
from paddle.fluid.executor import global_scope
from paddle.fluid.io import is_belong_to_optimizer
from paddle.fluid.dygraph.base import to_variable

from .callbacks import config_callbacks

__all__ = ['Model', 'Loss', 'CrossEntropy', 'Input']


class Input(fluid.dygraph.Layer):
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def forward(self):
        return fluid.data(self.name, shape=self.shape, dtype=self.dtype)


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def to_numpy(var):
    assert isinstance(var, (Variable, fluid.core.VarBase)), "not a variable"
    if isinstance(var, fluid.core.VarBase):
        return var.numpy()
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


class Loss(object):
    def __init__(self, average=True):
        super(Loss, self).__init__()
        self.average = average

    def forward(self, outputs, labels):
        raise NotImplementedError()

    def __call__(self, outputs, labels):
        labels = to_list(labels)
        if in_dygraph_mode():
            labels = [to_variable(l) for l in labels]
        losses = to_list(self.forward(to_list(outputs), labels))
        if self.average:
            losses = [fluid.layers.reduce_mean(l) for l in losses]
        else:
            losses = [fluid.layers.reduce_sum(l) for l in losses]
        return losses


class CrossEntropy(Loss):
    def __init__(self, average=True):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        return [
            fluid.layers.cross_entropy(o, l) for o, l in zip(outputs, labels)
        ]


class StaticGraphAdapter(object):
    def __init__(self, model):
        super(StaticGraphAdapter, self).__init__()
        self.model = model
        # with `_build_once` gone, parameters are now created in `__init__`
        # so we need to keep track of the parameters already created
        self._startup_prog = fluid.default_startup_program()
        self._orig_prog = fluid.default_main_program()

        self._label_vars = {}  # label variables
        self._input_vars = {}  # input variables
        self._endpoints = {}
        self._loss_endpoint = None
        self._executor = None
        self._progs = {}
        self._compiled_progs = {}

        self._lazy_load_optimizer = None

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    def train(self, inputs, labels=None, device='CPU', device_ids=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        self.mode = 'train'
        return self._run(inputs, labels, device, device_ids)

    def eval(self, inputs, labels=None, device='CPU', device_ids=None):
        self.mode = 'eval'
        return self._run(inputs, labels, device, device_ids)

    def test(self, inputs, device='CPU', device_ids=None):
        self.mode = 'test'
        return self._run(inputs, None, device, device_ids)

    def _compile(self, mode, device):
        self.mode = mode
        if self._progs.get(mode, None) is None:
            if self.model._inputs is None:
                raise ValueError("The inputs of Model must be not None.")
            input_vars = [k.forward() for k in to_list(self.model._inputs)]
            self._input_vars[mode] = input_vars
            self._make_program(input_vars)
        compiled_prog = self._compile_and_initialize(self._progs[mode], device)
        return compiled_prog

    def parameters(self, *args, **kwargs):
        return None

    def save(self, path):
        def _save(state, path):
            if not state:
                return
            state = {
                k: to_numpy(v) if isinstance(v, Variable) else v
                for k, v in state.items()
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        base = os.path.basename(path)
        assert base != "", "path should be of 'dirname/filename' format"
        param_path = path + ".pdparams"
        _save(self.model.state_dict(), param_path)
        prog = self._progs.get('train', None)
        if prog is None or self.model._optimizer is None:
            return
        # XXX `optimizer.state_dict()` only work in dygraph mode
        optim_path = path + ".pdopt"
        optim = {
            p.name: p
            for p in filter(is_belong_to_optimizer, prog.list_vars())
        }
        if not optim:
            return
        # HACK this is contrived, optimizer state is not the same for
        # static/dynamic graph mode
        optim['__static_graph_only__'] = True
        _save(optim, optim_path)

    def load(self, path):
        def _load(path):
            if not os.path.exists(path):
                return
            with open(path, 'rb') as f:
                return pickle.load(f)

        param_path = path + ".pdparams"
        param_state = _load(param_path)
        assert param_state, "failed to load parameters, please check path"

        if self._executor is None:
            executor = fluid.Executor(fluid.CPUPlace())._default_executor
        else:
            executor = self._executor._default_executor

        fluid.core._create_loaded_parameter(
            list(self.model.state_dict().values()), global_scope(), executor)

        for key, var in self.model.state_dict().items():
            assert key in param_state, \
                "parameter [{}] is not found in model file [{}]".format(
                    key, param_path)
            self._set_var(var, param_state[key])

        # FIXME what if a different optimizer is used?
        if not self.model._optimizer:
            return
        optim_path = path + ".pdopt"
        optim_state = _load(optim_path)
        if optim_state is None:
            return
        assert '__static_graph_only__' in optim_state, \
            "optimizer saved in dygraph mode is not usable in static graph"

        if self._executor is not None:
            self._load_optimizer(optim_state)
        else:
            self._lazy_load_optimizer = optim_state

    def _load_optimizer(self, state):
        prog = self._progs.get('train', None)
        optim = list(filter(is_belong_to_optimizer, prog.list_vars()))
        if not optim:
            return

        fluid.core._create_loaded_parameter(optim,
                                            global_scope(),
                                            self._executor._default_executor)

        for var in optim:
            assert var.name in state, \
                "variable [{}] is not in optimizer state file".format(var.name)
            self._set_var(var, state[var.name])

    def _set_var(self, var, ndarray):
        t = global_scope().find_var(var.name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = fluid.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = fluid.CUDAPinnedPlace()
        else:
            p = fluid.core.Place()
            p.set_place(t._place())
            place = fluid.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    def _run(self, inputs, labels=None, device='CPU', device_ids=None):
        inputs = to_list(inputs)
        labels = to_list(labels)
        assert len(inputs) == len(self.model._inputs), "number of inputs" \
            + " does not match number of arguments of `forward` method, " \
            + "{} != {}".format(len(inputs), len(self.model._inputs))

        compiled_prog = self._compile(self.mode, device)

        feed = {}
        input_names = [v.name for v in self._input_vars[self.mode]]
        for idx, n in enumerate(input_names):
            # train and test may take different arguments
            if inputs[idx] is not None:
                feed[n] = inputs[idx]
        if labels is not None:
            for idx, v in enumerate(self._label_vars[self.mode]):
                feed[v.name] = labels[idx]

        endpoints = self._endpoints[self.mode]
        fetch_list = endpoints['output']
        if 'loss' in endpoints:
            fetch_list = endpoints['output'] + endpoints['loss']

        num_output = len(endpoints['output'])
        out = self._executor.run(compiled_prog,
                                 feed=feed,
                                 fetch_list=fetch_list)
        if self.mode == 'test':
            return out[:num_output]
        else:
            return out[:num_output], out[num_output:]

    def _make_program(self, inputs):
        prog = self._orig_prog.clone()
        if self.mode == 'train' and self.model._optimizer._learning_rate_map:
            # HACK workaround learning rate map issue
            lr_var = self.model._optimizer._learning_rate_map[self._orig_prog]
            self.model._optimizer._learning_rate_map[prog] = lr_var
        losses = []
        with fluid.program_guard(prog, self._startup_prog):
            outputs = to_list(self.model.forward(*inputs))
            if self.mode != 'test':
                losses = self._get_loss(outputs)
                if self.mode == 'train':
                    self._loss_endpoint = fluid.layers.sum(losses)
                    self.model._optimizer.minimize(self._loss_endpoint)
        if self.mode != 'train':  # clone again to put it in test mode
            prog = prog.clone(for_test=True)
        self._progs[self.mode] = prog
        self._endpoints[self.mode] = {
            "output": outputs,
            "loss": losses
        } if self.model._loss_function else {
            'output': outputs
        }

    def _get_loss(self, outputs):
        if self.model._loss_function and self.model._loss:
            raise ValueError(
                "Do not set loss by model.set_loss() and "
                "loss_function in model.prepare() at the same time.")
        if self.model._loss_function is not None:
            if self.model._labels is None:
                raise ValueError("The labels of Model must be not None.")
            label_vars = [k.forward() for k in to_list(self.model._labels)]
            self._label_vars[self.mode] = label_vars
            losses = self.model._loss_function(outputs, label_vars)
        else:
            assert self.model._loss
            losses = to_list(self.model._loss)
        return losses

    def _compile_and_initialize(self, prog, device='CPU', device_ids=None):
        compiled_prog = self._compiled_progs.get(self.mode, None)
        if compiled_prog is not None:
            return compiled_prog

        device_ids = device_ids if device_ids else [0]
        places = [
            device.lower() == 'gpu' and fluid.CUDAPlace(i) or fluid.CPUPlace()
            for i in device_ids
        ]

        # XXX *ALL WEIGHTS* should be initialized upon model construction
        # even if `forward()` may run different code path for different mode
        # therefore startup program only needs to run once
        if self._executor is None:
            self._executor = fluid.Executor(places[0])
            # XXX incremental initialization
            uninitialized = []
            for var_py in self._startup_prog.list_vars():
                var = fluid.global_scope().find_var(var_py.name)
                if var and var.get_tensor()._is_initialized():
                    continue
                uninitialized.append(var_py)
            if uninitialized:
                startup_prog = self._startup_prog._prune(uninitialized)
                self._executor.run(startup_prog)

            if self.mode == 'train' and self._lazy_load_optimizer:
                self._load_optimizer(self._lazy_load_optimizer)
                self._lazy_load_optimizer = None

        compiled_prog = fluid.CompiledProgram(prog)
        #if len(device_ids) > 1:
        loss_name = None
        if self.mode == 'train' and self._loss_endpoint is not None:
            loss_name = self._loss_endpoint.name

        share_vars_from = None
        if self.mode == 'eval' and 'train' in self._compiled_progs:
            share_vars_from = self._compiled_progs['train']
        # HACK invalidate eval program if is compiled before train program
        # quite hackish, OTOH, it is generally uncommon that the eval
        # program will be run before the train program
        if self.mode == 'train' and 'eval' in self._compiled_progs:
            del self._compiled_progs['eval']

        compiled_prog = compiled_prog.with_data_parallel(
            loss_name=loss_name,
            #    places=places,
            share_vars_from=share_vars_from)

        self._compiled_progs[self.mode] = compiled_prog
        return compiled_prog


class DynamicGraphAdapter(object):
    def __init__(self, model):
        super(DynamicGraphAdapter, self).__init__()
        self.model = model

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    # TODO multi device in dygraph mode not implemented at present time
    def train(self, inputs, labels=None, device='CPU', device_ids=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        super(Model, self.model).train()
        self.mode = 'train'
        inputs = to_list(inputs)
        labels = to_list(labels)
        outputs = self.model.forward(*[to_variable(x) for x in inputs])
        losses = self._get_loss(outputs, labels)
        final_loss = fluid.layers.sum(losses)
        final_loss.backward()
        self.model._optimizer.minimize(final_loss)
        self.model.clear_gradients()
        return [to_numpy(o) for o in to_list(outputs)], \
            [to_numpy(l) for l in losses]

    def eval(self, inputs, labels=None, device='CPU', device_ids=None):
        super(Model, self.model).eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        outputs = self.model.forward(*[to_variable(x) for x in inputs])
        losses = self._get_loss(outputs, labels)
        return [to_numpy(o) for o in to_list(outputs)], \
            [to_numpy(l) for l in losses]

    def test(self, inputs, device='CPU', device_ids=None):
        super(Model, self.model).eval()
        self.mode = 'test'
        inputs = [to_variable(x) for x in to_list(inputs)]
        outputs = self.model.forward(*inputs)
        return [to_numpy(o) for o in to_list(outputs)]

    def _get_loss(self, outputs, labels):
        if self.model._loss_function and self.model._loss:
            raise ValueError(
                "Do not set loss by model.set_loss() and "
                "loss_function in model.prepare() at the same time.")
        if self.model._loss_function is not None:
            return self.model._loss_function(outputs, labels)
        else:
            return to_list(self.model._loss)

    def parameters(self, *args, **kwargs):
        return super(Model, self.model).parameters(*args, **kwargs)

    def save(self, path):
        params = self.model.state_dict()
        fluid.save_dygraph(params, path)
        if self.model._optimizer is None:
            return
        if self.model._optimizer.state_dict():
            optim = self.model._optimizer.state_dict()
            fluid.save_dygraph(optim, path)

    def load(self, path):
        params, optim = fluid.load_dygraph(path)
        self.model.set_dict(params)
        if self.model._optimizer is None or optim is None:
            return
        self.model._optimizer.set_dict(optim)


class Model(fluid.dygraph.Layer):
    """
    FIXME: add more comments and usage

    Args:
        inputs (Input|list of Input|None): inputs, entry points of network,
            could be a Input layer of lits of Input layers, or None.
            For static graph, inputs must be set. For dynamic graph, it could
            be None.
        labels (Input|list of Input|None): labels, entry points of network,
            could be a Input layer of lits of Input layers, or None.
            For static graph, if set loss_function in Model.prepare(), it
            must be set. Otherwise, it could be None.
    """

    def __init__(self, inputs=None, labels=None):
        super(Model, self).__init__(self.__class__.__name__)
        self.mode = 'train'
        self._inputs = to_list(inputs)
        self._labels = to_list(labels)
        self._loss_function = None
        self._loss_weights = None
        self._loss = None
        self._optimizer = None
        if in_dygraph_mode():
            self._adapter = DynamicGraphAdapter(self)
        else:
            self._adapter = StaticGraphAdapter(self)

    def train(self, *args, **kwargs):
        return self._adapter.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self._adapter.eval(*args, **kwargs)

    def test(self, *args, **kwargs):
        return self._adapter.test(*args, **kwargs)

    def fit(
            self,
            train_iterator=None,
            eval_iterator=None,
            eval_freq=1,
            epochs=1,
            device=None,
            verbose=2,
            callbacks=None, ):
        if device is None:
            device = 'GPU' if fluid.is_compiled_with_cuda() else 'CPU'
        do_eval = eval_iterator is not None
        cbks = config_callbacks(
            model=self,
            epochs=epochs,
            steps=None,
            verbose=verbose,
            metrics=['loss'],  # FIXME: more general
        )

        cbks.on_train_begin()
        train_logs = {}
        for epoch in range(epochs):
            cbks.on_epoch_begin(epoch)
            for step, data in enumerate(train_iterator):
                cbks.on_train_batch_begin(step)
                if not self._loss:
                    inputs, labels = data[0], data[1]
                else:
                    inputs, labels = data, None
                outs, loss = self.train(inputs, labels, device)
                train_logs['loss'] = np.sum(loss)
                cbks.on_train_batch_end(step, train_logs)
            cbks.on_epoch_end(epoch, train_logs)

            if do_eval and epoch % eval_freq == 0:
                params = {
                    'eval_steps': None,
                    'eval_metrics': ['eval_loss'],
                    'log_freq': 10
                }
                eval_logs = {}
                cbks.on_eval_begin(params)
                for step, data in enumerate(eval_iterator):
                    cbks.on_eval_batch_begin(step)

                    if not self._loss:
                        inputs, labels = data[0], data[1]
                    else:
                        inputs, labels = data, None
                    outs, loss = self.eval(inputs, labels, device)

                    eval_logs['eval_loss'] = np.sum(loss)
                    eval_logs['batch_size'] = inputs.shape[0]
                    cbks.on_eval_batch_end(step, eval_logs)
                cbks.on_eval_end(eval_logs)
        cbks.on_train_end(eval_logs)

    def save(self, *args, **kwargs):
        return self._adapter.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self._adapter.load(*args, **kwargs)

    def prepare(self, optimizer, loss_function=None):
        self._optimizer = optimizer
        if loss_function:
            if not isinstance(loss_function, Loss):
                raise TypeError(
                    "'loss_function' must be sub classes of 'Loss'")
        self._loss_function = loss_function

    def parameters(self, *args, **kwargs):
        return self._adapter.parameters(*args, **kwargs)

    def set_loss(self, loss):
        if loss and self._loss_function:
            raise ValueError(
                "Do not set loss by model.set_loss() and "
                "loss_function in model.prepare() at the same time.")
        if not isinstance(loss, (Variable, fluid.core.VarBase)):
            raise TypeError("loss type should be a Variable or VarBase.")
        self._loss = loss
