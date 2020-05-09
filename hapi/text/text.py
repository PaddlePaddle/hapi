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
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import ast
import time
import argparse as argparse
import numpy as np
import multiprocessing

import collections
import copy
from functools import partial, reduce

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.utils as utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.dygraph import to_variable, Embedding, Linear, LayerNorm, GRUUnit
from paddle.fluid.data_feeder import convert_dtype

from paddle.fluid import layers
from paddle.fluid.dygraph import Layer
from paddle.fluid.layers import BeamSearchDecoder

__all__ = [
    'RNNCell',
    'BasicLSTMCell',
    'BasicGRUCell',
    'RNN',
    'StackedLSTMCell',
    'LSTM',
    'BidirectionalLSTM',
    'StackedGRUCell',
    'GRU',
    'BidirectionalGRU',
    'DynamicDecode',
    'BeamSearchDecoder',
    'MultiHeadAttention',
    'FFN',
    'TransformerEncoderLayer',
    'TransformerEncoder',
    'TransformerDecoderLayer',
    'TransformerDecoder',
    'TransformerCell',
    'TransformerBeamSearchDecoder',
    'LinearChainCRF',
    'CRFDecoding',
    'SequenceTagging',
    'GRUEncoder',
]


class RNNCell(Layer):
    """
    RNNCell is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        """
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possibly nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possibly nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
            batch_dim_idx: An integer indicating which dimension of the tensor in
                inputs represents batch size.  The default value is 0.

        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(
                        lambda flag, x: isinstance(x, integer_types) and flag,
                        seq, True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape),
                                      states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: fluid.layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class BasicLSTMCell(RNNCell):
    """
    Long-Short Term Memory(LSTM) RNN cell.

    The formula used is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_
    for more details.

    Parameters:
        input_size (int): The input size in the LSTM cell.
        hidden_size (int): The hidden size in the LSTM cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias(float, optional): forget bias used when computing forget gate.
            Default 1.0
        dtype(string, optional): The data type used in this cell. Default float32.
        forget_gate_weights (dict, optional): A dict includes `w`, `h` and `b`
            as keys, and the corresponding values should be instances of Parameter
            which represent :math:`W_{x_{f}}, W_{h_{f}}, b_{f}` and have shape
            [input_size, hidden_size], [hidden_size, hidden_size], [hidden_size]
            separately. It is used for reusing and sharing weights when provided,
            otherwise create these parameters. Note that parameters from input
            gate, forget gate and cell would be concatenated in implementation.
        input_gate_weights (dict, optional): A dict includes `w`, `h` and `b` as keys,
            and the corresponding values should be instances of Parameter which
            represent :math:`W_{x_{i}}, W_{h_{i}}, b_{i}` separately. It has the
            same usage as :attr:`forget_gate_weights`.
        output_gate_weights (dict, optional): A dict includes `w`, `h` and `b` as keys,
            and the corresponding values should be instances of Parameter which
            represent :math:`W_{x_{o}}, W_{h_{o}}, b_{o}` separately. It has the
            same usage as :attr:`forget_gate_weights`.
        cell_weights (dict, optional): A dict includes `w`, `h` and `b` as keys,
            and the corresponding values should be instances of Parameter which
            represent :math:`W_{x_{c}}, W_{h_{c}}, b_{c}` separately. It has the
            same usage as :attr:`forget_gate_weights`.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32',
                 forget_gate_weights={"w": None,
                                      "h": None,
                                      "b": None},
                 input_gate_weights={"w": None,
                                     "h": None,
                                     "b": None},
                 output_gate_weights={"w": None,
                                      "h": None,
                                      "b": None},
                 cell_weights={"w": None,
                               "h": None,
                               "b": None}):
        super(BasicLSTMCell, self).__init__()

        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        # TODO(guosheng): find better way to resolve constants in __init__
        self._forget_bias = layers.create_global_var(
            shape=[1], dtype=dtype, value=forget_bias, persistable=True)
        self._forget_bias.stop_gradient = True
        self._dtype = dtype
        self._input_size = input_size

        self.use_customized_weight = False
        for _weights in [
                forget_gate_weights, input_gate_weights, output_gate_weights,
                cell_weights
        ]:
            for _key in _weights:
                if _weights[_key] is not None:
                    self.use_customized_weight = True
                    break
            if self.use_customized_weight:
                break

        if not self.use_customized_weight:

            self._weight = self.create_parameter(
                attr=self._param_attr,
                shape=[
                    self._input_size + self._hidden_size, 4 * self._hidden_size
                ],
                dtype=self._dtype)

            self._bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[4 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)
        else:
            if "w" in forget_gate_weights and forget_gate_weights[
                    "w"] is not None:
                self.fg_w = forget_gate_weights["w"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_forget_gate_w"
                else:
                    tmp_param_attr = self._param_attr
                self.fg_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in forget_gate_weights and forget_gate_weights[
                    "h"] is not None:
                self.fg_h = forget_gate_weights["h"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_forget_gate_h"
                else:
                    tmp_param_attr = self._param_attr
                self.fg_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in forget_gate_weights and forget_gate_weights[
                    "b"] is not None:
                self.fg_b = forget_gate_weights["b"]
            else:
                if self._bias_attr is not None and self._bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._bias_attr)
                    tmp_param_attr.name += "_forget_gate_b"
                else:
                    tmp_param_attr = self._bias_attr
                self.fg_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

            if "w" in input_gate_weights and input_gate_weights[
                    "w"] is not None:
                self.ig_w = input_gate_weights["w"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_input_gate_w"
                else:
                    tmp_param_attr = self._param_attr

                self.ig_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in input_gate_weights and input_gate_weights[
                    "h"] is not None:
                self.ig_h = input_gate_weights["h"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_input_gate_h"
                else:
                    tmp_param_attr = self._param_attr

                self.ig_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in input_gate_weights and input_gate_weights[
                    "b"] is not None:
                self.ig_b = input_gate_weights["b"]
            else:
                if self._bias_attr is not None and self._bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._bias_attr)
                    tmp_param_attr.name += "_input_gate_b"
                else:
                    tmp_param_attr = self._bias_attr
                self.ig_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

            if "w" in output_gate_weights and output_gate_weights[
                    "w"] is not None:
                self.og_w = output_gate_weights["w"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_output_gate_w"
                else:
                    tmp_param_attr = self._param_attr
                self.og_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in output_gate_weights and output_gate_weights[
                    "h"] is not None:
                self.og_h = output_gate_weights["h"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_output_gate_h"
                else:
                    tmp_param_attr = self._param_attr

                self.og_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in output_gate_weights and output_gate_weights[
                    "b"] is not None:
                self.og_b = output_gate_weights["b"]
            else:
                if self._bias_attr is not None and self._bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._bias_attr)
                    tmp_param_attr.name += "_output_gate_b"
                else:
                    tmp_param_attr = self._bias_attr
                self.og_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

            if "w" in cell_weights and cell_weights["w"] is not None:
                self.c_w = cell_weights["w"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_cell_w"
                else:
                    tmp_param_attr = self._param_attr

                self.c_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in cell_weights and cell_weights["h"] is not None:
                self.c_h = cell_weights["h"]
            else:
                if self._param_attr is not None and self._param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._param_attr)
                    tmp_param_attr.name += "_cell_h"
                else:
                    tmp_param_attr = self._param_attr
                self.c_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in cell_weights and cell_weights["b"] is not None:
                self.c_b = cell_weights["b"]
            else:
                if self._bias_attr is not None and self._bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(self._bias_attr)
                    tmp_param_attr.name += "_cell_b"
                else:
                    tmp_param_attr = self._bias_attr
                self.c_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

    def forward(self, input, state):

        if self.use_customized_weight:
            weight_w = fluid.layers.concat(
                [self.ig_w, self.c_w, self.fg_w, self.og_w], axis=-1)
            weight_h = fluid.layers.concat(
                [self.ig_h, self.c_h, self.fg_h, self.og_h], axis=-1)
            _weight = fluid.layers.concat([weight_w, weight_h], axis=0)
            _bias = fluid.layers.concat(
                [self.ig_b, self.c_b, self.fg_b, self.og_b])
        else:
            _weight = self._weight
            _bias = self._bias

        pre_hidden, pre_cell = state
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=_weight)

        gate_input = layers.elementwise_add(gate_input, _bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                self._gate_activation(
                    layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(
                self._gate_activation(i), self._activation(j)))
        new_hidden = self._activation(new_cell) * self._gate_activation(o)

        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicLSTMCell is a list with two shapes: `[[hidden_size], [hidden_size]]`
        (-1 for batch size would be automatically inserted into shape). These two
        shapes correspond to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return [[self._hidden_size], [self._hidden_size]]


class BasicGRUCell(RNNCell):
    """
    Gated Recurrent Unit (GRU) RNN cell.

    The formula for GRU used is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_
    for more details.

    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        gate_activation (function, optional): The activation function for gates
            of GRU, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            GRU, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        dtype(string, optional): The data type used in this cell. Default float32.
        update_gate_weights (dict, optional): A dict includes `w`, `h` and `b`
            as keys, and the corresponding values should be instances of Parameter
            which represent :math:`W_{ux}, W_{uh}, b_{u}` and have shape
            [input_size, hidden_size], [hidden_size, hidden_size], [hidden_size]
            separately. It is used for reusing and sharing weights when provided,
            otherwise create these parameters. Note that parameters from update
            gate and reset gate would be concatenated in implementation.
        reset_gate_weights (dict, optional): A dict includes `w`, `h` and `b` as keys,
            and the corresponding values should be instances of Parameter which
            represent :math:`W_{rx}, W_{rh}, b_{r}` separately. It has the
            same usage as :attr:`update_gate_weights`.
        cell_weights (dict, optional): A dict includes `w`, `h` and `b` as keys,
            and the corresponding values should be instances of Parameter which
            represent :math:`W_{cx}, W_{ch}, b_{c}`` separately. It has the
            same usage as :attr:`update_gate_weights`.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 dtype='float32',
                 update_gate_weights={"w": None,
                                      "h": None,
                                      "b": None},
                 reset_gate_weights={"w": None,
                                     "h": None,
                                     "b": None},
                 cell_weights={"w": None,
                               "h": None,
                               "b": None}):
        super(BasicGRUCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

        assert isinstance(update_gate_weights, dict)
        assert isinstance(reset_gate_weights, dict)
        assert isinstance(cell_weights, dict)

        self.use_customized_weight = False
        for _weights in [
                update_gate_weights, reset_gate_weights, cell_weights
        ]:
            for _key in _weights:
                if _weights[_key] is not None:
                    self.use_customized_weight = True
            if self.use_customized_weight:
                break

        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        if not self.use_customized_weight:
            self._gate_weight = self.create_parameter(
                attr=gate_param_attr,
                shape=[
                    self._input_size + self._hidden_size, 2 * self._hidden_size
                ],
                dtype=self._dtype)

            self._candidate_weight = self.create_parameter(
                attr=candidate_param_attr,
                shape=[
                    self._input_size + self._hidden_size, self._hidden_size
                ],
                dtype=self._dtype)

            if self._bias_attr is not None and self._bias_attr.name is not None:
                gate_bias_attr = copy.deepcopy(self._bias_attr)
                candidate_bias_attr = copy.deepcopy(self._bias_attr)
                gate_bias_attr.name += "_gate"
                candidate_bias_attr.name += "_candidate"
            else:
                gate_bias_attr = self._bias_attr
                candidate_bias_attr = self._bias_attr

            self._gate_bias = self.create_parameter(
                attr=gate_bias_attr,
                shape=[2 * self._hidden_size],
                dtype=self._dtype,
                is_bias=True)
            self._candidate_bias = self.create_parameter(
                attr=candidate_bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        else:

            # create the parameters of gates in gru
            if "w" in update_gate_weights and update_gate_weights[
                    "w"] is not None:
                self.ug_w = update_gate_weights["w"]
            else:
                if gate_param_attr is not None and gate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_param_attr)
                    tmp_param_attr.name += "_update_gate_w"
                else:
                    tmp_param_attr = gate_param_attr
                self.ug_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in update_gate_weights and update_gate_weights[
                    "h"] is not None:
                self.ug_h = update_gate_weights["h"]
            else:
                if gate_param_attr is not None and gate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_param_attr)
                    tmp_param_attr.name += "_update_gate_h"
                else:
                    tmp_param_attr = gate_param_attr
                self.ug_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in update_gate_weights and update_gate_weights[
                    "b"] is not None:
                self.ug_b = update_gate_weights["b"]
            else:
                if gate_bias_attr is not None and gate_bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_bias_attr)
                    tmp_param_attr.name += "_update_gate_b"
                else:
                    tmp_param_attr = gate_bias_attr
                self.ug_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

            # reset gate parameters
            if "w" in reset_gate_weights and reset_gate_weights[
                    "w"] is not None:
                self.rg_w = reset_gate_weights["w"]
            else:
                if gate_param_attr is not None and gate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_param_attr)
                    tmp_param_attr.name += "_reset_gate_w"
                else:
                    tmp_param_attr = gate_param_attr
                self.rg_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in reset_gate_weights and reset_gate_weights[
                    "h"] is not None:
                self.rg_h = reset_gate_weights["h"]
            else:
                if gate_param_attr is not None and gate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_param_attr)
                    tmp_param_attr.name += "_reset_gate_h"
                else:
                    tmp_param_attr = gate_param_attr
                self.rg_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in reset_gate_weights and reset_gate_weights[
                    "b"] is not None:
                self.rg_b = reset_gate_weights["b"]
            else:
                if gate_bias_attr is not None and gate_bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(gate_bias_attr)
                    tmp_param_attr.name += "_reset_gate_b"
                else:
                    tmp_param_attr = gate_bias_attr
                self.rg_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

            # cell parameters
            if "w" in cell_weights and cell_weights["w"] is not None:
                self.c_w = cell_weights["w"]
            else:
                if candidate_param_attr is not None and candidate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(candidate_param_attr)
                    tmp_param_attr.name += "_cell_w"
                else:
                    tmp_param_attr = gate_param_attr

                self.c_w = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._input_size, self._hidden_size],
                    dtype=self._dtype)

            if "h" in cell_weights and cell_weights["h"] is not None:
                self.c_h = cell_weights["h"]
            else:
                if candidate_param_attr is not None and candidate_param_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(candidate_param_attr)
                    tmp_param_attr.name += "_cell_h"
                else:
                    tmp_param_attr = gate_param_attr
                self.c_h = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size, self._hidden_size],
                    dtype=self._dtype)

            if "b" in cell_weights and cell_weights["b"] is not None:
                self.c_b = cell_weights["b"]
            else:
                if candidate_bias_attr is not None and candidate_bias_attr.name is not None:
                    tmp_param_attr = copy.deepcopy(candidate_bias_attr)
                    tmp_param_attr.name += "_cell_b"
                else:
                    tmp_param_attr = gate_bias_attr
                self.c_b = self.create_parameter(
                    attr=tmp_param_attr,
                    shape=[self._hidden_size],
                    dtype=self._dtype,
                    is_bias=True)

    def forward(self, input, state):

        if self.use_customized_weight:
            rg_weights = layers.concat([self.rg_w, self.rg_h], axis=0)
            ug_weights = layers.concat([self.ug_w, self.ug_h], axis=0)
            _gate_weight = layers.concat([rg_weights, ug_weights], axis=-1)
            _candidate_weight = layers.concat([self.c_w, self.c_h], axis=0)
            _gate_bias = layers.concat([self.rg_b, self.ug_b], axis=0)
            _candidate_bias = self.c_b
        else:
            _gate_weight = self._gate_weight
            _gate_bias = self._gate_bias
            _candidate_weight = self._candidate_weight
            _candidate_bias = self._candidate_bias

        pre_hidden = state
        concat_input_hidden = layers.concat([input, pre_hidden], axis=1)

        gate_input = layers.matmul(x=concat_input_hidden, y=_gate_weight)

        gate_input = layers.elementwise_add(gate_input, _gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden], 1), _candidate_weight)
        candidate = layers.elementwise_add(candidate, _candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden, new_hidden

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicGRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return [self._hidden_size]


class RNN(Layer):
    """
    RNN creates a recurrent neural network specified by RNNCell `cell`, which
    performs :code:`cell.forward()` repeatedly until reaches to the maximum
    length of `inputs`.

    Parameters:
        cell(RNNCell): An instance of `RNNCell`.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell, inputs=inputs)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major
        self.batch_index, self.time_step_index = (1, 0) if time_major else (0,
                                                                            1)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        """
        Performs :code:`cell.forward()` repeatedly until reaches to the maximum
        length of `inputs`.

        Parameters:
            inputs (Variable): A (possibly nested structure of) tensor variable[s]. 
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be unrolled
                in RNN.
            initial_states (Variable, optional): A (possibly nested structure of)
                tensor variable[s], representing the initial state for RNN. 
                If not provided, `cell.get_initial_states` would be used to produce
                the initial state. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
            **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`. 

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
                outputs and states, both are Tensor or nested structure of Tensor. \
                `final_outputs` has the same structure and data types as \
                the returned `outputs` of :code:`cell.forward` , and each Tenser in `final_outputs` \
                stacks all time steps' counterpart in `outputs` thus has shape `[batch_size, sequence_length, ...]` \
                for `time_major == False` or `[sequence_length, batch_size, ...]` for `time_major == True`. \
                `final_states` is the counterpart at last time step of initial states, \
                thus has the same structure with it and has tensors with same shapes \
                and data types.
        """
        if fluid.in_dygraph_mode():

            class ArrayWrapper(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)
                    return self

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                new_state = fluid.layers.elementwise_mul(
                    new_state, step_mask,
                    axis=0) - fluid.layers.elementwise_mul(
                        state, (step_mask - 1), axis=0)
                return new_state

            flat_inputs = flatten(inputs)
            batch_size, time_steps = (
                flat_inputs[0].shape[self.batch_index],
                flat_inputs[0].shape[self.time_step_index])

            if initial_states is None:
                initial_states = self.cell.get_initial_states(
                    batch_ref=inputs, batch_dim_idx=self.batch_index)

            if not self.time_major:
                inputs = map_structure(
                    lambda x: fluid.layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = fluid.layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = fluid.layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=[0]), inputs)
                mask = fluid.layers.reverse(
                    mask, axis=[0]) if sequence_length is not None else None

            states = initial_states
            outputs = []
            for i in range(time_steps):
                step_inputs = map_structure(lambda x: x[i], inputs)
                step_outputs, new_states = self.cell(step_inputs, states,
                                                     **kwargs)
                if sequence_length is not None:
                    new_states = map_structure(
                        partial(
                            _maybe_copy, step_mask=mask[i]),
                        states,
                        new_states)
                states = new_states
                outputs = map_structure(
                    lambda x: ArrayWrapper(x),
                    step_outputs) if i == 0 else map_structure(
                        lambda x, x_array: x_array.append(x), step_outputs,
                        outputs)

            final_outputs = map_structure(
                lambda x: fluid.layers.stack(x.array,
                                             axis=self.time_step_index),
                outputs)

            if self.is_reverse:
                final_outputs = map_structure(
                    lambda x: fluid.layers.reverse(x,
                                                   axis=self.time_step_index),
                    final_outputs)

            final_states = new_states
        else:
            final_outputs, final_states = fluid.layers.rnn(
                self.cell,
                inputs,
                initial_states=initial_states,
                sequence_length=sequence_length,
                time_major=self.time_major,
                is_reverse=self.is_reverse,
                **kwargs)
        return final_outputs, final_states


class StackedRNNCell(RNNCell):
    """
    Wrapper allowing a stack of RNN cells to behave as a single cell. It is used
    to implement stacked RNNs.

    Parameters:
        cells (list|tuple): List of RNN cell instances.

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.text import BasicLSTMCell, StackedRNNCell

            cells = [BasicLSTMCell(32, 32), BasicLSTMCell(32, 32)]
            stack_rnn = StackedRNNCell(cells)
    """

    def __init__(self, cells):
        self.cells = []
        for i, cell in enumerate(cells):
            self.cells.append(self.add_sublayer("cell_%d" % i, cell))

    def forward(self, inputs, states, **kwargs):
        """
        Performs :code:`cell.forward` for all including cells sequentially.
        Each cell's `inputs` is the `outputs` of the previous cell. And each
        cell's `states` is the corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. Mostly it is a
                float32 or float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ). `outputs` is the \
                `outputs` of the last cell. `new_states` is a list composed \
                of all cells' `new_states`, and its structure and data type is \
                same as that of `states` argument.
        """
        new_states = []
        for cell, state in zip(self.cells, states):
            outputs, new_state = cell(inputs, state, **kwargs)
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @staticmethod
    def stack_param_attr(param_attr, n):
        """
        If `param_attr` is a list or tuple, convert every element in it to a
        ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
        construct a list, and rename every one by appending a increasing index
        suffix to avoid having same names when `param_attr` contains a name.

        Parameters:
            param_attr (list|tuple|ParamAttr): A list, tuple or something can be
                converted to a ParamAttr instance by `ParamAttr._to_attr`.
            n (int): The times to repeat to construct a list when `param_attr`
                is not a list or tuple.

        Returns:
            list: A list composed of each including cell's `param_attr`.
        """
        if isinstance(param_attr, (list, tuple)):
            assert len(param_attr) == n, (
                "length of param_attr should be %d when it is a list/tuple" %
                n)
            param_attrs = [
                fluid.ParamAttr._to_attr(attr) for attr in param_attr
            ]
        else:
            param_attrs = []
            attr = fluid.ParamAttr._to_attr(param_attr)
            for i in range(n):
                attr_i = copy.deepcopy(attr)
                if attr.name:
                    attr_i.name = attr_i.name + "_" + str(i)
                param_attrs.append(attr_i)
        return param_attrs

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedRNNCell is a list composed of each including
        cell's `state_shape`.

        Returns:
            list: A list composed of each including cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class StackedLSTMCell(RNNCell):
    """
    Wrapper allowing a stack of LSTM cells to behave as a single cell. It is used
    to implement stacked LSTM.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input size for the first LSTM cell.
        hidden_size (int): The hidden size for every LSTM cell.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias (float, optional): forget bias used when computing forget
            gate. It also can accept a boolean value `True`, which would set
            :math:`forget\\_bias` as 0 but initialize :math:`b_{f}` as 1 and
            :math:`b_{i}, b_{f}, b_{c}, b_{0}` as 0. This is recommended in
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf .
            Default 1.0.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell, inputs=inputs)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedLSTMCell, self).__init__()
        self.dropout = utils.convert_to_list(dropout, num_layers, "dropout",
                                             float)
        param_attrs = StackedRNNCell.stack_param_attr(param_attr, num_layers)
        bias_attrs = StackedRNNCell.stack_param_attr(bias_attr, num_layers)

        self.cells = []
        for i in range(num_layers):
            if forget_bias is True:
                bias_attrs[
                    i].initializer = fluid.initializer.NumpyArrayInitializer(
                        np.concatenate(
                            np.zeros(2 * hidden_size),
                            np.ones(hidden_size), np.zeros(hidden_size))
                        .astype(dtype))
                forget_bias = 0.0
            self.cells.append(
                self.add_sublayer(
                    "lstm_%d" % i,
                    BasicLSTMCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        gate_activation=gate_activation,
                        activation=activation,
                        forget_bias=forget_bias,
                        param_attr=param_attrs[i],
                        bias_attr=bias_attrs[i],
                        dtype=dtype)))

    def forward(self, inputs, states):
        """
        Performs the stacked LSTM cells sequentially. Each cell's `inputs` is
        the `outputs` of the previous cell. And each cell's `states` is the
        corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. It is a float32 or
                float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula of the last LSTM; `new_states` \
                is a list composed of every LSTM `new_states` which is a pair \
                of tensors standing for :math:`h_{t}, c_{t}` in the formula, \
                and the data type and structure of these tensors all is same \
                as that of `states`.
        """
        new_states = []
        for i, cell in enumerate(self.cells):
            outputs, new_state = cell(inputs, states[i])
            outputs = layers.dropout(
                outputs,
                self.dropout[i],
                dropout_implementation='upscale_in_train') if self.dropout[
                    i] > 0 else outputs
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedLSTMCell is a list composed of each including
        LSTM cell's `state_shape`.

        Returns:
            list: A list composed of each including LSTM cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class LSTM(Layer):
    """
    Applies a stacked multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input feature size for the first LSTM.
        hidden_size (int): The hidden size for every LSTM.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias (float, optional): forget bias used when computing forget
            gate. It also can accept a boolean value `True`, which would set
            :math:`forget\\_bias` as 0 but initialize :math:`b_{f}` as 1 and
            :math:`b_{i}, b_{f}, b_{c}, b_{0}` as 0. This is recommended in
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf .
            Default 1.0.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import LSTM

            inputs = paddle.rand((2, 4, 32))
            lstm = LSTM(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = lstm(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 is_reverse=False,
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(LSTM, self).__init__()
        lstm_cell = StackedLSTMCell(input_size, hidden_size, gate_activation,
                                    activation, forget_bias, num_layers,
                                    dropout, param_attr, bias_attr, dtype)
        self.lstm = RNN(lstm_cell, is_reverse, time_major)

    def forward(self, inputs, initial_states=None, sequence_length=None):
        """
        Performs the stacked multi-layer LSTM layer by layer. Each LSTM's `outputs`
        is the `inputs` of the subsequent one.

        Parameters:
            inputs (Variable): The inputs for the first LSTM. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked LSTM, and the initial states of each LSTM is a pair
                of tensors shaped `[batch_size, hidden_size]`. If not provided,
                use 0 as initial states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last LSTM and it is a tensor with shape \
                `[batch_size, sequence_length, hidden_size]` and has the same \
                data type as `inputs`, `final_states` is the counterpart of \
                `initial_states` at last time step, thus has the same structure \
                with it and has tensors with same shapes data types. 
        """
        return self.lstm(inputs, initial_states, sequence_length)


class BidirectionalRNN(Layer):
    """
    Wrapper for bidirectional RNN. It assembles two RNNCell instances to perform
    forward and backward RNN separately, and merge outputs of these two RNN
    according to `merge_mode`.

    Parameters:
        cell_fw (RNNCell): A RNNCell instance used for forward RNN.
        cell_bw (RNNCell): A RNNCell instance used for backward RNN.
        merge_mode (str|None, optional): The way to merget outputs of forward and
            backward RNN. It can be `concat`, `sum`, `ave`, `mul`, `zip` and None,
            where None stands for make the two `outputs` as a tuple, `zip` stands
            for make each two corresponding tensors of the two `outputs` as a tuple.
            Default `concat`

    Examples:

        .. code-block:: python

            import paddle
            from paddle.incubate.hapi.text import BasicLSTMCell, StackedRNNCell

            inputs = paddle.rand((2, 4, 32))
            cell_fw = StackedLSTMCell(32, 64)
            cell_bw = StackedLSTMCell(32, 64)
            bi_rnn = BidirectionalRNN(cell_fw, cell_bw)
            outputs, _ = bi_rnn(inputs)  # [2, 4, 128]
    """

    def __init__(self,
                 cell_fw,
                 cell_bw,
                 merge_mode='concat',
                 time_major=False,
                 cell_cls=None,
                 **kwargs):
        super(BidirectionalRNN, self).__init__()
        self.rnn_fw = RNN(cell_fw, is_reverse=False, time_major=time_major)
        self.rnn_bw = RNN(cell_bw, is_reverse=True, time_major=time_major)
        if merge_mode == 'concat':
            self.merge_func = lambda x, y: layers.concat([x, y], -1)
        elif merge_mode == 'sum':
            self.merge_func = lambda x, y: layers.elementwise_add(x, y)
        elif merge_mode == 'ave':
            self.merge_func = lambda x, y: layers.scale(
                layers.elementwise_add(x, y), 0.5)
        elif merge_mode == 'mul':
            self.merge_func = lambda x, y: layers.elementwise_mul(x, y)
        elif merge_mode == 'zip':
            self.merge_func = lambda x, y: (x, y)
        elif merge_mode is None:
            self.merge_func = None
        else:
            raise ValueError('Unsupported value for `merge_mode`: %s' %
                             merge_mode)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        """
        Performs forward and backward RNN separately, and merge outputs of these
        two RNN according to `merge_mode`.

        Parameters:
            inputs (Variable): A (possibly nested structure of) tensor variable[s]. 
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be unrolled
                in both forward and backward RNN.
            initial_states (Variable|list|tuple): If it is a list or tuple, its
                length should be 2 to include initial states of forward and backward
                RNN separately. Otherwise it would be used twice for the two RNN. 
                If None, `cell.get_initial_states` would be used to produce the initial
                states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
            **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is produced by merge outputs of forward and backward RNN according \
                to `merge_mode`, `final_states` is a pair including `final_states` \
                of forward and backward RNN.
        """
        if isinstance(initial_states, (list, tuple)):
            assert len(
                initial_states
            ) == 2, "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]
        outputs_fw, states_fw = self.rnn_fw(inputs, initial_states[0],
                                            sequence_length, **kwargs)
        outputs_bw, states_bw = self.rnn_bw(inputs, initial_states[1],
                                            sequence_length, **kwargs)
        outputs = map_structure(
            self.merge_func, outputs_fw,
            outputs_bw) if self.merge_func else (outputs_fw, outputs_bw)
        return outputs, (states_fw, states_bw)

    @staticmethod
    def bidirect_param_attr(param_attr):
        """
        Converts `param_attr` to a pair of `param_attr` when it is not a list
        or tuple with length 2, also rename every one by appending a suffix to
        avoid having same names when `param_attr` contains a name.

        Parameters:
            param_attr (list|tuple|ParamAttr): A list, tuple or something can be
                converted to a ParamAttr instance by `ParamAttr._to_attr`. When
                it is a list or tuple, its length must be 2.

        Returns:
            list: A pair composed of forward and backward RNN cell's `param_attr`.
        """
        if isinstance(param_attr, (list, tuple)):
            assert len(
                param_attr
            ) == 2, "length of param_attr should be 2 when it is a list/tuple"
            param_attrs = param_attr
        else:
            param_attrs = []
            attr = fluid.ParamAttr._to_attr(param_attr)
            attr_fw = copy.deepcopy(attr)
            if attr.name:
                attr_fw.name = attr_fw.name + "_fw"
            param_attrs.append(attr_fw)
            attr_bw = copy.deepcopy(attr)
            if attr.name:
                attr_bw.name = attr_bw.name + "_bw"
            param_attrs.append(attr_bw)
        return param_attrs


class BidirectionalLSTM(Layer):
    """
    Applies a bidirectional multi-layer long short-term memory (LSTM) RNN to an
    input sequence. 
    
    Bidirection interaction can happen after each layer or only after the last
    layer according to the  `merge_each_layer` setting. The way to interact,
    that is how to merge outputs of the two direction, is determined by `merge_mode`.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input feature size for the first LSTM.
        hidden_size (int): The hidden size for every LSTM.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        forget_bias (float, optional): forget bias used when computing forget
            gate. It also can accept a boolean value `True`, which would set
            :math:`forget\\_bias` as 0 but initialize :math:`b_{f}` as 1 and
            :math:`b_{i}, b_{f}, b_{c}, b_{0}` as 0. This is recommended in
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf .
            Default 1.0.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        merge_mode (str|None, optional): The way to merget outputs of forward and
            backward RNN. It can be `concat`, `sum`, `ave`, `mul`, `zip` and None,
            where None stands for make the two `outputs` as a tuple, `zip` stands
            for make each two corresponding tensors of the two `outputs` as a tuple.
            Default `concat`
        merge_each_layer (bool, optional): Indicate whether bidirection interaction
            happens after each layer or only after the last layer. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import BidirectionalLSTM

            inputs = paddle.rand((2, 4, 32))
            bi_lstm = BidirectionalLSTM(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = bi_lstm(inputs)  # [2, 4, 128]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 merge_mode='concat',
                 merge_each_layer=False,
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(BidirectionalLSTM, self).__init__()
        self.num_layers = num_layers
        self.merge_mode = merge_mode
        self.merge_each_layer = merge_each_layer
        param_attrs = BidirectionalRNN.bidirect_param_attr(param_attr)
        bias_attrs = BidirectionalRNN.bidirect_param_attr(bias_attr)
        if not merge_each_layer:
            cell_fw = StackedLSTMCell(input_size, hidden_size, gate_activation,
                                      activation, forget_bias, num_layers,
                                      dropout, param_attrs[0], bias_attrs[0],
                                      dtype)
            cell_bw = StackedLSTMCell(input_size, hidden_size, gate_activation,
                                      activation, forget_bias, num_layers,
                                      dropout, param_attrs[1], bias_attrs[1],
                                      dtype)
            self.lstm = BidirectionalRNN(
                cell_fw, cell_bw, merge_mode=merge_mode, time_major=time_major)
        else:
            fw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[0],
                                                             num_layers)
            bw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[1],
                                                             num_layers)
            fw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[0],
                                                            num_layers)
            bw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[1],
                                                            num_layers)

            # maybe design cell including both forward and backward later
            self.lstm = []
            for i in range(num_layers):
                cell_fw = StackedLSTMCell(
                    input_size if i == 0 else (hidden_size * 2
                                               if merge_mode == 'concat' else
                                               hidden_size), hidden_size,
                    gate_activation, activation, forget_bias, 1, dropout,
                    fw_param_attrs[i], fw_bias_attrs[i], dtype)
                cell_bw = StackedLSTMCell(
                    input_size if i == 0 else (hidden_size * 2
                                               if merge_mode == 'concat' else
                                               hidden_size), hidden_size,
                    gate_activation, activation, forget_bias, 1, dropout,
                    bw_param_attrs[i], bw_bias_attrs[i], dtype)
                self.lstm.append(
                    self.add_sublayer(
                        "lstm_%d" % i,
                        BidirectionalRNN(
                            cell_fw,
                            cell_bw,
                            merge_mode=merge_mode,
                            time_major=time_major)))

    def forward(self, inputs, initial_states=None, sequence_length=None):
        """
        Performs bidirectional multi-layer LSTM layer by layer. Each LSTM's `outputs`
        is the `inputs` of the subsequent one, or when `merge_each_layer` is True,
        merged outputs would be the `inputs` of the subsequent one.

        Parameters:
            inputs (Variable): The inputs for the first LSTM. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked LSTM. If `merge_each_layer` is True, the length of
                list should be `num_layers` and a single value would be reused for
                `num_layers`; Otherwise, the length should be 2 and a single value
                would be reused twice. If not provided, use 0 as initial states.
                Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last bidirectional LSTM; `final_states` is a \
                pair including `final_states` of forward and backward LSTM when \
                `merge_each_layer` is False or a list including `final_states` \
                of all stacked bidirectional LSTM, and it has tensors with same \
                shapes data types as `initial_states`.
        """
        if not self.merge_each_layer:
            return self.lstm(inputs, initial_states, sequence_length)
        else:
            if isinstance(initial_states, (list, tuple)):
                assert len(initial_states) == self.num_layers, (
                    "length of initial_states should be %d when it is a list/tuple"
                    % self.num_layers)
            else:
                initial_states = [initial_states] * self.num_layers
            stacked_states = []
            for i in range(self.num_layers):
                outputs, states = self.lstm[i](inputs, initial_states[i],
                                               sequence_length)
                inputs = outputs
                stacked_states.append(states)
            return outputs, stacked_states


class StackedGRUCell(RNNCell):
    """
    Wrapper allowing a stack of GRU cells to behave as a single cell. It is used
    to implement stacked GRU.

    The formula for GRU used here is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        gate_activation (function, optional): The activation function for gates
            of GRU, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            GRU, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            GRU. It also can be a list or tuple, including dropout probabilities
            for the corresponding GRU. Default 0.0
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedGRUCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell, inputs=inputs)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedGRUCell, self).__init__()
        self.dropout = utils.convert_to_list(dropout, num_layers, "dropout",
                                             float)
        param_attrs = StackedRNNCell.stack_param_attr(param_attr, num_layers)
        bias_attrs = StackedRNNCell.stack_param_attr(bias_attr, num_layers)

        self.cells = []
        for i in range(num_layers):
            self.cells.append(
                self.add_sublayer(
                    "gru_%d" % i,
                    BasicGRUCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        gate_activation=gate_activation,
                        activation=activation,
                        param_attr=param_attrs[i],
                        bias_attr=bias_attrs[i],
                        dtype=dtype)))

    def forward(self, inputs, states):
        """
        Performs the stacked GRU cells sequentially. Each cell's `inputs` is
        the `outputs` of the previous cell. And each cell's `states` is the
        corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. It is a float32 or
                float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula of the last GRU; `new_states` \
                is a list composed of every GRU `new_states` which is also \
                :math:`h_{t}` in the formula, and the data type and structure \
                of these tensors all is same as that of `states`.
        """
        new_states = []
        for i, cell in enumerate(self.cells):
            outputs, new_state = cell(inputs, states[i])
            outputs = layers.dropout(
                outputs,
                self.dropout[i],
                dropout_implementation='upscale_in_train') if self.dropout[
                    i] > 0 else outputs
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedGRUCell is a list composed of each including
        GRU cell's `state_shape`.

        Returns:
            list: A list composed of each including GRU cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class GRU(Layer):
    """
    Applies a stacked multi-layer gated recurrent unit (GRU) RNN to an input
    sequence.

    The formula for GRU used here is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        gate_activation (function, optional): The activation function for gates
            of GRU, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            GRU, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        num_layers(int, optional): The number of GRU to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            GRU. It also can be a list or tuple, including dropout probabilities
            for the corresponding GRU. Default 0.0
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import LSTM

            inputs = paddle.rand((2, 4, 32))
            gru = GRU(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = gru(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 num_layers=1,
                 dropout=0.0,
                 is_reverse=False,
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(GRU, self).__init__()
        gru_cell = StackedGRUCell(input_size, hidden_size, gate_activation,
                                  activation, num_layers, dropout, param_attr,
                                  bias_attr, dtype)
        self.gru = RNN(gru_cell, is_reverse, time_major)

    def forward(self, inputs, initial_states=None, sequence_length=None):
        """
        Performs the stacked multi-layer GRU layer by layer. Each GRU's `outputs`
        is the `inputs` of the subsequent one.

        Parameters:
            inputs (Variable): The inputs for the first GRU. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked GRU, and the initial states of each GRU is a tensor
                shaped `[batch_size, hidden_size]`. If not provided, use 0 as initial
                states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last GRU and it is a tensor with shape \
                `[batch_size, sequence_length, hidden_size]` and has the same \
                data type as `inputs`, `final_states` is the counterpart of \
                `initial_states` at last time step, thus has the same structure \
                with it and has tensors with same shapes data types.
        """
        return self.gru(inputs, initial_states, sequence_length)


class BidirectionalGRU(Layer):
    """
    Applies a bidirectional multi-layer gated recurrent unit (GRU) RNN to an input
    sequence.
    
    Bidirection interaction can happen after each layer or only after the last
    layer according to the  `merge_each_layer` setting. The way to interact,
    that is how to merge outputs of the two direction, is determined by `merge_mode`.

    The formula for GRU used here is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        gate_activation (function, optional): The activation function for gates
            of GRU, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            GRU, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        num_layers(int, optional): The number of GRU to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            GRU. It also can be a list or tuple, including dropout probabilities
            for the corresponding GRU. Default 0.0
        merge_mode (str|None, optional): The way to merget outputs of forward and
            backward RNN. It can be `concat`, `sum`, `ave`, `mul`, `zip` and None,
            where None stands for make the two `outputs` as a tuple, `zip` stands
            for make each two corresponding tensors of the two `outputs` as a tuple.
            Default `concat`
        merge_each_layer (bool, optional): Indicate whether bidirection interaction
            happens after each layer or only after the last layer. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import BidirectionalGRU

            inputs = paddle.rand((2, 4, 32))
            gru = BidirectionalGRU(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = bi_gru(inputs)  # [2, 4, 128]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 num_layers=1,
                 dropout=0.0,
                 merge_mode='concat',
                 merge_each_layer=False,
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(BidirectionalGRU, self).__init__()
        self.num_layers = num_layers
        self.merge_mode = merge_mode
        self.merge_each_layer = merge_each_layer
        param_attrs = BidirectionalRNN.bidirect_param_attr(param_attr)
        bias_attrs = BidirectionalRNN.bidirect_param_attr(bias_attr)
        if not merge_each_layer:
            cell_fw = StackedGRUCell(input_size, hidden_size, gate_activation,
                                     activation, num_layers, dropout,
                                     param_attrs[0], bias_attrs[0], dtype)
            cell_bw = StackedGRUCell(input_size, hidden_size, gate_activation,
                                     activation, num_layers, dropout,
                                     param_attrs[1], bias_attrs[1], dtype)
            self.gru = BidirectionalRNN(
                cell_fw, cell_bw, merge_mode=merge_mode, time_major=time_major)
        else:
            fw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[0],
                                                             num_layers)
            bw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[1],
                                                             num_layers)
            fw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[0],
                                                            num_layers)
            bw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[1],
                                                            num_layers)

            # maybe design cell including both forward and backward later
            self.gru = []
            for i in range(num_layers):
                cell_fw = StackedGRUCell(input_size if i == 0 else (
                    hidden_size * 2 if merge_mode == 'concat' else
                    hidden_size), hidden_size, gate_activation, activation, 1,
                                         dropout, fw_param_attrs[i],
                                         fw_bias_attrs[i], dtype)
                cell_bw = StackedGRUCell(input_size if i == 0 else (
                    hidden_size * 2 if merge_mode == 'concat' else
                    hidden_size), hidden_size, gate_activation, activation, 1,
                                         dropout, bw_param_attrs[i],
                                         bw_bias_attrs[i], dtype)
                self.gru.append(
                    self.add_sublayer(
                        "gru_%d" % i,
                        BidirectionalRNN(
                            cell_fw,
                            cell_bw,
                            merge_mode=merge_mode,
                            time_major=time_major)))

    def forward(self, inputs, initial_states=None, sequence_length=None):
        """
        Performs bidirectional multi-layer GRU layer by layer. Each GRU's `outputs`
        is the `inputs` of the subsequent one, or when `merge_each_layer` is True,
        merged outputs would be the `inputs` of the subsequent one.

        Parameters:
            inputs (Variable): The inputs for the first GRU. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked GRU. If `merge_each_layer` is True, the length of
                list should be `num_layers` and a single value would be reused for
                `num_layers`; Otherwise, the length should be 2 and a single value
                would be reused twice. If not provided, use 0 as initial states.
                Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.

        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last bidirectional GRU; `final_states` is a \
                pair including `final_states` of forward and backward GRU when \
                `merge_each_layer` is False or a list including `final_states` \
                of all stacked bidirectional GRU, and it has tensors with same \
                shapes data types as `initial_states`.
        """
        if not self.merge_each_layer:
            return self.gru(inputs, initial_states, sequence_length)
        else:
            if isinstance(initial_states, (list, tuple)):
                assert len(initial_states) == self.num_layers, (
                    "length of initial_states should be %d when it is a list/tuple"
                    % self.num_layers)
            else:
                initial_states = [initial_states] * self.num_layers
            stacked_states = []
            for i in range(self.num_layers):
                outputs, states = self.gru[i](inputs, initial_states[i],
                                              sequence_length)
                inputs = outputs
                stacked_states.append(states)
            return outputs, stacked_states


class DynamicDecode(Layer):
    """
    DynamicDecode integrates an Decoder instance to perform dynamic decoding.

    It performs :code:`decoder.step()` repeatedly until the returned Tensor
    indicating finished status contains all True values or the number of
    decoding step reaches to :attr:`max_step_num`.

    :code:`decoder.initialize()` would be called once before the decoding loop.
    If the `decoder` has implemented `finalize` method, :code:`decoder.finalize()`
    would be called once after the decoding loop.

    Parameters:
        decoder (Decoder): An instance of `Decoder`.
        max_step_num (int, optional): The maximum number of steps. If not provided,
            decode until the decoder is fully done, or in other words, the returned
            Tensor by :code:`decoder.step()` indicating finished status contains
            all True. Default `None`.
        output_time_major (bool, optional): Indicate the data layout of Tensor included
            in the final outputs(the first returned value of this method). If
            attr:`False`, the data layout would be batch major with shape
            `[batch_size, seq_len, ...]`.  If attr:`True`, the data layout would
            be time major with shape `[seq_len, batch_size, ...]`. Default: `False`.
        impute_finished (bool, optional): If `True`, then states get copied through
            for batch entries which are marked as finished, which differs with the
            unfinished using the new states returned by :code:`decoder.step()` and
            ensures that the final states have the correct values. Otherwise, states
            wouldn't be copied through when finished. If the returned `final_states`
            is needed, it should be set as True, which causes some slowdown.
            Default `False`.
        is_test (bool, optional): A flag indicating whether to use test mode. In
            test mode, it is more memory saving. Default `False`.
        return_length (bool, optional):  A flag indicating whether to return an
            extra Tensor variable in the output tuple, which stores the actual
            lengths of all decoded sequences. Default `False`.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            vocab_size, d_model, = 100, 32
            encoder_output = paddle.rand((2, 4, d_model))
            trg_embeder = fluid.dygraph.Embedding(size=[vocab_size, d_model])
            output_layer = fluid.dygraph.Linear(d_model, vocab_size)
            cell = StackedLSTMCell(input_size=d_model, hidden_size=d_model)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)
            dynamic_decoder = DynamicDecode(decoder, max_step_num=10)
            outputs = dynamic_decoder(cell.get_initial_states(encoder_output))
    """

    def __init__(self,
                 decoder,
                 max_step_num=None,
                 output_time_major=False,
                 impute_finished=False,
                 is_test=False,
                 return_length=False):
        super(DynamicDecode, self).__init__()
        self.decoder = decoder
        self.max_step_num = max_step_num
        self.output_time_major = output_time_major
        self.impute_finished = impute_finished
        self.is_test = is_test
        self.return_length = return_length

    def forward(self, inits=None, **kwargs):
        """
        Performs :code:`decoder.step()` repeatedly until the returned Tensor
        indicating finished status contains all True values or the number of
        decoding step reaches to :attr:`max_step_num`.

        :code:`decoder.initialize()` would be called once before the decoding loop.
        If the `decoder` has implemented `finalize` method, :code:`decoder.finalize()`
        would be called once after the decoding loop.

        Parameters:
            inits (object, optional): Argument passed to `decoder.initialize`.
                Default `None`.
            **kwargs: Additional keyword arguments. Arguments passed to `decoder.step`.

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states, sequence_lengths)` ) \
                when `return_length` is True, otherwise a tuple( :code:`(final_outputs, final_states)` ). \
                The final outputs and states, both are Tensor or nested structure of Tensor. \
                `final_outputs` has the same structure and data types as the :code:`outputs` \
                returned by :code:`decoder.step()` , and each Tenser in `final_outputs` \
                is the stacked of all decoding steps' outputs, which might be revised \
                by :code:`decoder.finalize()` if the decoder has implemented `finalize`. \
                `final_states` is the counterpart at last time step of initial states \
                returned by :code:`decoder.initialize()` , thus has the same structure \
                with it and has tensors with same shapes and data types. `sequence_lengths` \
                is an `int64` tensor with the same shape as `finished` returned \
                by :code:`decoder.initialize()` , and it stores the actual lengths of \
                all decoded sequences.
        """
        if fluid.in_dygraph_mode():

            class ArrayWrapper(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)
                    return self

                def __getitem__(self, item):
                    return self.array.__getitem__(item)

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                state_dtype = state.dtype
                if convert_dtype(state_dtype) in ["bool"]:
                    state = layers.cast(state, dtype="float32")
                    new_state = layers.cast(new_state, dtype="float32")
                if step_mask.dtype != state.dtype:
                    step_mask = layers.cast(step_mask, dtype=state.dtype)
                    # otherwise, renamed bool gradients of would be summed up leading
                    # to sum(bool) error.
                    step_mask.stop_gradient = True
                new_state = layers.elementwise_mul(
                    state, step_mask, axis=0) - layers.elementwise_mul(
                        new_state, (step_mask - 1), axis=0)
                if convert_dtype(state_dtype) in ["bool"]:
                    new_state = layers.cast(new_state, dtype=state_dtype)
                return new_state

            initial_inputs, initial_states, initial_finished = self.decoder.initialize(
                inits)
            inputs, states, finished = (initial_inputs, initial_states,
                                        initial_finished)
            cond = layers.logical_not((layers.reduce_all(initial_finished)))
            sequence_lengths = layers.cast(
                layers.zeros_like(initial_finished), "int64")
            outputs = None

            step_idx = 0
            step_idx_tensor = layers.fill_constant(
                shape=[1], dtype="int64", value=step_idx)
            while cond.numpy():
                (step_outputs, next_states, next_inputs,
                 next_finished) = self.decoder.step(step_idx_tensor, inputs,
                                                    states, **kwargs)
                if not self.decoder.tracks_own_finished:
                    # BeamSearchDecoder would track it own finished, since
                    # beams would be reordered and the finished status of each
                    # entry might change. Otherwise, perform logical OR which
                    # would not change the already finished.
                    next_finished = layers.logical_or(next_finished, finished)
                    # To confirm states.finished/finished be consistent with
                    # next_finished.
                    layers.assign(next_finished, finished)
                next_sequence_lengths = layers.elementwise_add(
                    sequence_lengths,
                    layers.cast(
                        layers.logical_not(finished), sequence_lengths.dtype))

                if self.impute_finished:  # rectify the states for the finished.
                    next_states = map_structure(
                        lambda x, y: _maybe_copy(x, y, finished), states,
                        next_states)
                outputs = map_structure(
                    lambda x: ArrayWrapper(x),
                    step_outputs) if step_idx == 0 else map_structure(
                        lambda x, x_array: x_array.append(x), step_outputs,
                        outputs)
                inputs, states, finished, sequence_lengths = (
                    next_inputs, next_states, next_finished,
                    next_sequence_lengths)

                layers.increment(x=step_idx_tensor, value=1.0, in_place=True)
                step_idx += 1

                layers.logical_not(layers.reduce_all(finished), cond)
                if self.max_step_num is not None and step_idx > self.max_step_num:
                    break

            final_outputs = map_structure(
                lambda x: fluid.layers.stack(x.array, axis=0), outputs)
            final_states = states

            try:
                final_outputs, final_states = self.decoder.finalize(
                    final_outputs, final_states, sequence_lengths)
            except NotImplementedError:
                pass

            if not self.output_time_major:
                final_outputs = map_structure(
                    lambda x: layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), final_outputs)

            return (final_outputs, final_states,
                    sequence_lengths) if self.return_length else (
                        final_outputs, final_states)
        else:
            return fluid.layers.dynamic_decode(
                self.decoder,
                inits,
                max_step_num=self.max_step_num,
                output_time_major=self.output_time_major,
                impute_finished=self.impute_finished,
                is_test=self.is_test,
                return_length=self.return_length,
                **kwargs)


class TransformerCell(Layer):
    """
    Let inputs=(trg_word, trg_pos), states=cache to make Transformer can be
    used as RNNCell
    """

    def __init__(self, decoder, embedding_fn=None, output_fn=None):
        super(TransformerCell, self).__init__()
        self.decoder = decoder
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn

    def forward(self, inputs, states, trg_src_attn_bias, enc_output,
                static_caches):
        trg_word, trg_pos = inputs
        for cache, static_cache in zip(states, static_caches):
            cache.update(static_cache)
        if self.embedding_fn is not None:
            dec_input = self.embedding_fn(trg_word, trg_pos)
            outputs = self.decoder(dec_input, enc_output, None,
                                   trg_src_attn_bias, states)
        else:
            outputs = self.decoder(trg_word, trg_pos, enc_output, None,
                                   trg_src_attn_bias, states)
        if self.output_fn is not None:
            outputs = self.output_fn(outputs)
        if len(outputs.shape) == 3:
            # squeeze to adapt to BeamSearchDecoder which use 2D logits
            outputs = layers.squeeze(outputs, [1])
        new_states = [{"k": cache["k"], "v": cache["v"]} for cache in states]
        return outputs, new_states

    @property
    def state_shape(self):
        return [{
            "k": [self.decoder.n_head, 0, self.decoder.d_key],
            "v": [self.decoder.n_head, 0, self.decoder.d_value],
        } for i in range(len(self.decoder.n_layer))]


class TransformerBeamSearchDecoder(layers.BeamSearchDecoder):
    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, x):
        # init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        x = layers.transpose(x,
                             list(range(var_dim_in_state, len(x.shape))) +
                             list(range(0, var_dim_in_state)))
        x = layers.reshape(
            x, [0] * (len(x.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
            [int(size) for size in x.shape[-var_dim_in_state + 2:]])
        x = layers.transpose(
            x,
            list(range((len(x.shape) + 1 - var_dim_in_state), len(x.shape))) +
            list(range(0, (len(x.shape) + 1 - var_dim_in_state))))
        return x

    def _split_batch_beams_with_var_dim(self, x):
        var_dim_size = layers.shape(x)[self.var_dim_in_state]
        x = layers.reshape(
            x, [-1, self.beam_size] +
            [int(size)
             for size in x.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in x.shape[self.var_dim_in_state + 1:]])
        return x

    def step(self, time, inputs, states, **kwargs):
        # compared to RNN, Transformer has 3D data at every decoding step
        inputs = layers.reshape(inputs, [-1, 1])  # token
        pos = layers.ones_like(inputs) * time  # pos
        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)
        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)


### Transformer Modules ###
class PrePostProcessLayer(Layer):
    """
    PrePostProcessLayer
    """

    def __init__(self,
                 process_cmd,
                 d_model,
                 dropout_rate,
                 reused_layer_norm=None):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = []
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                self.functors.append(
                    lambda x, y: x + y if y is not None else x)
            elif cmd == "n":  # add layer normalization
                if reused_layer_norm is not None:
                    layer_norm = reused_layer_norm
                else:
                    layer_norm = LayerNorm(
                        normalized_shape=d_model,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(1.)),
                        bias_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(0.)))

                self.functors.append(
                    self.add_sublayer(
                        "layer_norm_%d" % len(
                            self.sublayers(include_sublayers=False)),
                        layer_norm))
            elif cmd == "d":  # add dropout
                self.functors.append(lambda x: layers.dropout(
                    x, dropout_prob=dropout_rate, is_test=False)
                                     if dropout_rate else x)

    def forward(self, x, residual=None):
        for i, cmd in enumerate(self.process_cmd):
            if cmd == "a":
                x = self.functors[i](x, residual)
            else:
                x = self.functors[i](x)
        return x


class MultiHeadAttention(Layer):
    """
    Multi-Head Attention
    """

    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.0,
                 reused_query_fc=None,
                 reused_key_fc=None,
                 reused_value_fc=None,
                 reused_proj_fc=None):

        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        if reused_query_fc is not None:
            self.q_fc = reused_query_fc
        else:
            self.q_fc = Linear(
                input_dim=d_model, output_dim=d_key * n_head, bias_attr=False)
        if reused_key_fc is not None:
            self.k_fc = reused_key_fc
        else:
            self.k_fc = Linear(
                input_dim=d_model, output_dim=d_key * n_head, bias_attr=False)
        if reused_value_fc is not None:
            self.v_fc = reused_value_fc
        else:
            self.v_fc = Linear(
                input_dim=d_model,
                output_dim=d_value * n_head,
                bias_attr=False)
        if reused_proj_fc is not None:
            self.proj_fc = reused_proj_fc
        else:
            self.proj_fc = Linear(
                input_dim=d_value * n_head,
                output_dim=d_model,
                bias_attr=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:  # self-attention
            keys, values = queries, queries
            static_kv = False
        else:  # cross-attention
            static_kv = True

        q = self.q_fc(queries)
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])

        if cache is not None and static_kv and "static_k" in cache:
            # for encoder-decoder attention in inference and has cached
            k = cache["static_k"]
            v = cache["static_v"]
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
            k = layers.transpose(x=k, perm=[0, 2, 1, 3])
            v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
            v = layers.transpose(x=v, perm=[0, 2, 1, 3])

        if cache is not None:
            if static_kv and not "static_k" in cache:
                # for encoder-decoder attention in inference and has not cached
                cache["static_k"], cache["static_v"] = k, v
            elif not static_kv:
                # for decoder self-attention in inference
                cache_k, cache_v = cache["k"], cache["v"]
                k = layers.concat([cache_k, k], axis=2)
                v = layers.concat([cache_v, v], axis=2)
                cache["k"], cache["v"] = k, v

        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(queries, keys, values, cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.d_model**-0.5)
        if attn_bias is not None:
            product += attn_bias
        weights = layers.softmax(product)
        if self.dropout_rate:
            weights = layers.dropout(
                weights, dropout_prob=self.dropout_rate, is_test=False)

        out = layers.matmul(weights, v)

        # combine heads
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.proj_fc(out)
        return out

    def cal_kv(self, keys, values):
        k = self.k_fc(keys)
        v = self.v_fc(values)
        k = layers.reshape(x=k, shape=[0, 0, self.n_head, self.d_key])
        k = layers.transpose(x=k, perm=[0, 2, 1, 3])
        v = layers.reshape(x=v, shape=[0, 0, self.n_head, self.d_value])
        v = layers.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v


class FFN(Layer):
    """
    Feed-Forward Network
    """

    def __init__(self,
                 d_inner_hid,
                 d_model,
                 dropout_rate,
                 fc1_act="relu",
                 reused_fc1=None,
                 reused_fc2=None):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        if reused_fc1 is not None:
            self.fc1 = reused_fc1
        else:
            self.fc1 = Linear(
                input_dim=d_model, output_dim=d_inner_hid, act=fc1_act)
        if reused_fc2 is not None:
            self.fc2 = reused_fc2
        else:
            self.fc2 = Linear(input_dim=d_inner_hid, output_dim=d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        if self.dropout_rate:
            hidden = layers.dropout(
                hidden, dropout_prob=self.dropout_rate, is_test=False)
        out = self.fc2(hidden)
        return out


class TransformerEncoderLayer(Layer):
    """
    EncoderLayer
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 ffn_fc1_act="relu",
                 reused_pre_selatt_layernorm=None,
                 reused_multihead_att_weights={
                     "reused_query_fc": None,
                     "reused_key_fc": None,
                     "reused_value_fc": None,
                     "reused_proj_fc": None
                 },
                 reused_post_selfatt_layernorm=None,
                 reused_pre_ffn_layernorm=None,
                 reused_ffn_weights={"reused_fc1": None,
                                     "reused_fc2": None},
                 reused_post_ffn_layernorm=None):

        super(TransformerEncoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout,
                                                 reused_pre_selatt_layernorm)
        self.self_attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            reused_query_fc=reused_multihead_att_weights["reused_query_fc"],
            reused_key_fc=reused_multihead_att_weights["reused_key_fc"],
            reused_value_fc=reused_multihead_att_weights["reused_value_fc"],
            reused_proj_fc=reused_multihead_att_weights["reused_proj_fc"])
        self.postprocesser1 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout,
            reused_post_selfatt_layernorm)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout,
                                                 reused_pre_ffn_layernorm)
        self.ffn = FFN(d_inner_hid,
                       d_model,
                       relu_dropout,
                       fc1_act=ffn_fc1_act,
                       reused_fc1=reused_ffn_weights["reused_fc1"],
                       reused_fc2=reused_ffn_weights["reused_fc2"])
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout,
                                                  reused_post_ffn_layernorm)

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(
            self.preprocesser1(enc_input), None, None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)

        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class TransformerEncoder(Layer):
    """
    encoder
    """

    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 ffn_fc1_act="relu"):

        super(TransformerEncoder, self).__init__()

        self.encoder_layers = list()
        for i in range(n_layer):
            self.encoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    TransformerEncoderLayer(
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        preprocess_cmd,
                        postprocess_cmd,
                        ffn_fc1_act=ffn_fc1_act)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output

        return self.processer(enc_output)


class TransformerDecoderLayer(Layer):
    """
    decoder
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 reused_pre_selfatt_layernorm=None,
                 reused_self_multihead_att_weights={
                     "reused_query_fc": None,
                     "reused_key_fc": None,
                     "reused_value_fc": None,
                     "reused_proj_fc": None
                 },
                 reused_post_selfatt_layernorm=None,
                 reused_pre_crossatt_layernorm=None,
                 reused_cross_multihead_att_weights={
                     "reused_query_fc": None,
                     "reused_key_fc": None,
                     "reused_value_fc": None,
                     "reused_proj_fc": None
                 },
                 reused_post_crossatt_layernorm=None,
                 reused_pre_ffn_layernorm=None,
                 reused_ffn_weights={"reused_fc1": None,
                                     "reused_fc2": None},
                 reused_post_ffn_layernorm=None):
        super(TransformerDecoderLayer, self).__init__()

        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout,
                                                 reused_pre_selfatt_layernorm)
        self.self_attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            reused_query_fc=reused_self_multihead_att_weights[
                "reused_query_fc"],
            reused_key_fc=reused_self_multihead_att_weights["reused_key_fc"],
            reused_value_fc=reused_self_multihead_att_weights[
                "reused_value_fc"],
            reused_proj_fc=reused_self_multihead_att_weights["reused_proj_fc"])
        self.postprocesser1 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout,
            reused_post_selfatt_layernorm)

        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout,
                                                 reused_pre_crossatt_layernorm)
        self.cross_attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            reused_query_fc=reused_cross_multihead_att_weights[
                "reused_query_fc"],
            reused_key_fc=reused_cross_multihead_att_weights["reused_key_fc"],
            reused_value_fc=reused_cross_multihead_att_weights[
                "reused_value_fc"],
            reused_proj_fc=reused_cross_multihead_att_weights[
                "reused_proj_fc"])
        self.postprocesser2 = PrePostProcessLayer(
            postprocess_cmd, d_model, prepostprocess_dropout,
            reused_post_crossatt_layernorm)

        self.preprocesser3 = PrePostProcessLayer(preprocess_cmd, d_model,
                                                 prepostprocess_dropout,
                                                 reused_pre_ffn_layernorm)
        self.ffn = FFN(d_inner_hid,
                       d_model,
                       relu_dropout,
                       reused_fc1=reused_ffn_weights["reused_fc1"],
                       reused_fc2=reused_ffn_weights["reused_fc2"])
        self.postprocesser3 = PrePostProcessLayer(postprocess_cmd, d_model,
                                                  prepostprocess_dropout,
                                                  reused_post_ffn_layernorm)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                cache=None):
        self_attn_output = self.self_attn(
            self.preprocesser1(dec_input), None, None, self_attn_bias, cache)
        self_attn_output = self.postprocesser1(self_attn_output, dec_input)

        cross_attn_output = self.cross_attn(
            self.preprocesser2(self_attn_output), enc_output, enc_output,
            cross_attn_bias, cache)
        cross_attn_output = self.postprocesser2(cross_attn_output,
                                                self_attn_output)

        ffn_output = self.ffn(self.preprocesser3(cross_attn_output))
        ffn_output = self.postprocesser3(ffn_output, cross_attn_output)

        return ffn_output


class TransformerDecoder(Layer):
    """
    decoder
    """

    def __init__(self, n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
                 prepostprocess_dropout, attention_dropout, relu_dropout,
                 preprocess_cmd, postprocess_cmd):
        super(TransformerDecoder, self).__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value

        self.decoder_layers = list()
        for i in range(n_layer):
            self.decoder_layers.append(
                self.add_sublayer(
                    "layer_%d" % i,
                    TransformerDecoderLayer(
                        n_head, d_key, d_value, d_model, d_inner_hid,
                        prepostprocess_dropout, attention_dropout,
                        relu_dropout, preprocess_cmd, postprocess_cmd)))
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
                                             prepostprocess_dropout)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_bias,
                cross_attn_bias,
                caches=None):
        for i, decoder_layer in enumerate(self.decoder_layers):
            dec_output = decoder_layer(dec_input, enc_output, self_attn_bias,
                                       cross_attn_bias, None
                                       if caches is None else caches[i])
            dec_input = dec_output

        return self.processer(dec_output)

    def prepare_static_cache(self, enc_output):
        return [
            dict(
                zip(("static_k", "static_v"),
                    decoder_layer.cross_attn.cal_kv(enc_output, enc_output)))
            for decoder_layer in self.decoder_layers
        ]

    def prepare_incremental_cache(self, enc_output):
        return [{
            "k": layers.fill_constant_batch_size_like(
                input=enc_output,
                shape=[-1, self.n_head, 0, self.d_key],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant_batch_size_like(
                input=enc_output,
                shape=[-1, self.n_head, 0, self.d_value],
                dtype=enc_output.dtype,
                value=0),
        } for i in range(self.n_layer)]


#TODO: we should merge GRUCell with BasicGRUCell
class GRUCell(RNNCell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.fc_layer = Linear(
            input_size, hidden_size * 3, param_attr=param_attr)

        self.gru_unit = GRUUnit(
            hidden_size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

    def forward(self, inputs, states):
        # for GRUCell, `step_outputs` and `new_states` both are hidden
        x = self.fc_layer(inputs)
        hidden, _, _ = self.gru_unit(x, states)
        return hidden, hidden

    @property
    def state_shape(self):
        return [self.hidden_size]


#TODO: we should merge GRUCell with BasicGRUCell
class GRUEncoderCell(RNNCell):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(GRUEncoderCell, self).__init__()
        self.dropout_prob = dropout_prob
        # use add_sublayer to add multi-layers
        self.gru_cells = []
        for i in range(num_layers):
            self.gru_cells.append(
                self.add_sublayer(
                    "gru_%d" % i,
                    #BasicGRUCell(
                    GRUCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.UniformInitializer(
                                low=-init_scale, high=init_scale)))))

    def forward(self, step_input, states):
        new_states = []
        for i, gru_cell in enumerate(self.gru_cells):
            out, state = gru_cell(step_input, states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation='upscale_in_train'
            ) if self.dropout_prob > 0 else out
            new_states.append(step_input)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.gru_cells]


class BiGRU(fluid.dygraph.Layer):
    def __init__(self, input_dim, grnn_hidden_dim, init_bound, h_0=None):
        super(BiGRU, self).__init__()
        self.gru = RNN(GRUEncoderCell(1, input_dim, grnn_hidden_dim, 0.0,
                                      init_bound),
                       is_reverse=False,
                       time_major=False)

        self.gru_r = RNN(GRUEncoderCell(1, input_dim, grnn_hidden_dim, 0.0,
                                        init_bound),
                         is_reverse=True,
                         time_major=False)

    def forward(self, input_feature):
        pre_gru, pre_state = self.gru(input_feature)
        gru_r, r_state = self.gru_r(input_feature)
        bi_merge = fluid.layers.concat(input=[pre_gru, gru_r], axis=-1)
        return bi_merge


class LinearChainCRF(Layer):
    def __init__(self, param_attr, size=None, is_test=False, dtype='float32'):
        super(LinearChainCRF, self).__init__()

        self._param_attr = param_attr
        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label, length=None):

        alpha = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        emission_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        transition_exps = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        log_likelihood = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": [label]
        }
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(
            type='linear_chain_crf',
            inputs=this_inputs,
            outputs={
                "Alpha": [alpha],
                "EmissionExps": [emission_exps],
                "TransitionExps": transition_exps,
                "LogLikelihood": log_likelihood
            },
            attrs={"is_test": self._is_test, })
        return log_likelihood


class CRFDecoding(Layer):
    def __init__(self, param_attr, size=None, is_test=False, dtype='float32'):
        super(CRFDecoding, self).__init__()

        self._dtype = dtype
        self._size = size
        self._is_test = is_test
        self._param_attr = param_attr
        self._transition = self.create_parameter(
            attr=self._param_attr,
            shape=[self._size + 2, self._size],
            dtype=self._dtype)

    @property
    def weight(self):
        return self._transition

    @weight.setter
    def weight(self, value):
        self._transition = value

    def forward(self, input, label=None, length=None):

        viterbi_path = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        this_inputs = {
            "Emission": [input],
            "Transition": self._transition,
            "Label": label
        }
        if length is not None:
            this_inputs['Length'] = [length]
        self._helper.append_op(
            type='crf_decoding',
            inputs=this_inputs,
            outputs={"ViterbiPath": [viterbi_path]},
            attrs={"is_test": self._is_test, })
        return viterbi_path


class GRUEncoder(Layer):
    def __init__(self,
                 input_dim,
                 grnn_hidden_dim,
                 init_bound,
                 num_layers=1,
                 is_bidirection=False):
        super(GRUEncoder, self).__init__()
        self.num_layers = num_layers
        self.is_bidirection = is_bidirection
        self.gru_list = []
        self.gru_r_list = []
        for i in range(num_layers):
            self.basic_gru_cell = BasicGRUCell(
                input_size=input_dim if i == 0 else input_dim * 2,
                hidden_size=grnn_hidden_dim,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-init_bound, high=init_bound),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            self.gru_list.append(
                self.add_sublayer(
                    "gru_%d" % i,
                    RNN(self.basic_gru_cell,
                        is_reverse=False,
                        time_major=False)))
        if self.is_bidirection:
            for i in range(num_layers):
                self.basic_gru_cell_r = BasicGRUCell(
                    input_size=input_dim if i == 0 else input_dim * 2,
                    hidden_size=grnn_hidden_dim,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.UniformInitializer(
                            low=-init_bound, high=init_bound),
                        regularizer=fluid.regularizer.L2DecayRegularizer(
                            regularization_coeff=1e-4)))
                self.gru_r_list.append(
                    self.add_sublayer(
                        "gru_r_%d" % i,
                        RNN(self.basic_gru_cell_r,
                            is_reverse=True,
                            time_major=False)))

    def forward(self, input_feature, h0=None):
        for i in range(self.num_layers):
            pre_gru, pre_state = self.gru_list[i](input_feature)
            if self.is_bidirection:
                gru_r, r_state = self.gru_r_list[i](input_feature)
                out = fluid.layers.concat(input=[pre_gru, gru_r], axis=-1)
            else:
                out = pre_gru
            input_feature = out
        return out


class SequenceTagging(Layer):
    def __init__(self,
                 vocab_size,
                 num_labels,
                 word_emb_dim=128,
                 grnn_hidden_dim=128,
                 emb_learning_rate=0.1,
                 crf_learning_rate=0.1,
                 bigru_num=2,
                 init_bound=0.1):
        super(SequenceTagging, self).__init__()
        """
        define the sequence tagging network structure
        word: stores the input of the model
        for_infer: a boolean value, indicating if the model to be created is for training or predicting.

        return:
            for infer: return the prediction
            otherwise: return the prediction
        """
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.grnn_hidden_dim = grnn_hidden_dim
        self.emb_lr = emb_learning_rate
        self.crf_lr = crf_learning_rate
        self.bigru_num = bigru_num
        self.init_bound = 0.1

        self.word_embedding = Embedding(
            size=[self.vocab_size, self.word_emb_dim],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                learning_rate=self.emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))

        self.gru_encoder = GRUEncoder(
            input_dim=self.grnn_hidden_dim,
            grnn_hidden_dim=self.grnn_hidden_dim,
            init_bound=self.init_bound,
            num_layers=self.bigru_num,
            is_bidirection=True)

        self.fc = Linear(
            input_dim=self.grnn_hidden_dim * 2,
            output_dim=self.num_labels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        self.linear_chain_crf = LinearChainCRF(
            param_attr=fluid.ParamAttr(
                name='linear_chain_crfw', learning_rate=self.crf_lr),
            size=self.num_labels)

        self.crf_decoding = CRFDecoding(
            param_attr=fluid.ParamAttr(
                name='crfw', learning_rate=self.crf_lr),
            size=self.num_labels)

    def forward(self, word, lengths, target=None):
        """
        Configure the network
        """
        word_embed = self.word_embedding(word)
        input_feature = word_embed

        bigru_output = self.gru_encoder(input_feature)
        emission = self.fc(bigru_output)

        if target is not None:
            crf_cost = self.linear_chain_crf(
                input=emission, label=target, length=lengths)
            avg_cost = fluid.layers.mean(x=crf_cost)
            self.crf_decoding.weight = self.linear_chain_crf.weight
            crf_decode = self.crf_decoding(input=emission, length=lengths)
            return crf_decode, avg_cost, lengths
        else:
            self.linear_chain_crf.weight = self.crf_decoding.weight
            crf_decode = self.crf_decoding(input=emission, length=lengths)
            return crf_decode, lengths
