import collections
import copy
import six
import sys
from functools import partial, reduce

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.utils as utils
from paddle.fluid.layers.utils import map_structure, flatten, pack_sequence_as
from paddle.fluid.dygraph import to_variable, Embedding, Linear, LayerNorm
from paddle.fluid.data_feeder import convert_dtype

from paddle.fluid import layers
from paddle.fluid.dygraph import Layer
from paddle.fluid.layers import BeamSearchDecoder

__all__ = [
    'RNNCell', 'BasicLSTMCell', 'BasicGRUCell', 'RNN', 'DynamicDecode',
    'BeamSearchDecoder', 'MultiHeadAttention', 'FFN',
    'TransformerEncoderLayer', 'TransformerEncoder', 'TransformerDecoderLayer',
    'TransformerDecoder', 'TransformerBeamSearchDecoder'
]


class RNNCell(Layer):
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
            shape: A (possiblely nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possiblely nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.

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
    ****
    BasicLSTMUnit class, Using basic operator to build LSTM
    The algorithm can be described as the code below.
        .. math::
           i_t &= \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_i)
           f_t &= \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_f + forget_bias )
           o_t &= \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_o)
           \\tilde{c_t} &= tanh(W_{cx}x_t + W_{ch}h_{t-1} + b_c)
           c_t &= f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}
           h_t &= o_t \odot tanh(c_t)
        - $W$ terms denote weight matrices (e.g. $W_{ix}$ is the matrix
          of weights from the input gate to the input)
        - The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
        - sigmoid is the logistic sigmoid function.
        - $i, f, o$ and $c$ are the input gate, forget gate, output gate,
          and cell activation vectors, respectively, all of which have the same size as
          the cell output activation vector $h$.
        - The :math:`\odot` is the element-wise product of the vectors.
        - :math:`tanh` is the activation functions.
        - :math:`\\tilde{c_t}` is also called candidate hidden state,
          which is computed based on the current input and the previous hidden state.
    Args:
        name_scope(string) : The name scope used to identify parameter and bias name
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of LSTM unit.
            If it is set to None or one attribute of ParamAttr, lstm_unit will
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized as zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cells (actNode).
                             Default: 'fluid.layers.tanh'
        forget_bias(float|1.0): forget bias used when computing forget gate
        dtype(string): data type used in this unit
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
        self._forget_bias = layers.fill_constant(
            [1], dtype=dtype, value=forget_bias)
        self._forget_bias.stop_gradient = False
        self._dtype = dtype
        self._input_size = input_size

        assert isinstance(forget_gate_weights, dict)
        assert isinstance(input_gate_weights, dict)
        assert isinstance(output_gate_weights, dict)
        assert isinstance(cell_weights, dict)

        # forgot get parameters
        if "w" in forget_gate_weights and forget_gate_weights["w"] is not None:
            self.fg_w = forget_gate_weights["w"]
        else:
            self.fg_w = self.create_parameter(
                attr=self._param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in forget_gate_weights and forget_gate_weights["h"] is not None:
            self.fg_h = forget_gate_weights["h"]
        else:
            self.fg_h = self.create_parameter(
                attr=self._param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in forget_gate_weights and forget_gate_weights["b"] is not None:
            self.fg_b = forget_gate_weights["b"]
        else:
            self.fg_b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # input gate parameters
        if "w" in input_gate_weights and input_gate_weights["w"] is not None:
            self.ig_w = input_gate_weights["w"]
        else:
            self.ig_w = self.create_parameter(
                attr=self._param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in input_gate_weights and input_gate_weights["h"] is not None:
            self.ig_h = input_gate_weights["h"]
        else:
            self.ig_h = self.create_parameter(
                attr=self._param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in input_gate_weights and input_gate_weights["b"] is not None:
            self.ig_b = input_gate_weights["b"]
        else:
            self.ig_b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # output gate parameters
        if "w" in output_gate_weights and output_gate_weights["w"] is not None:
            self.og_w = output_gate_weights["w"]
        else:
            self.og_w = self.create_parameter(
                attr=self._param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in output_gate_weights and output_gate_weights["h"] is not None:
            self.og_h = output_gate_weights["h"]
        else:
            self.og_h = self.create_parameter(
                attr=self._param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in output_gate_weights and output_gate_weights["b"] is not None:
            self.og_b = output_gate_weights["b"]
        else:
            self.og_b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # cell parameters
        if "w" in cell_weights and cell_weights["w"] is not None:
            self.c_w = cell_weights["w"]
        else:
            self.c_w = self.create_parameter(
                attr=self._param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in cell_weights and cell_weights["h"] is not None:
            self.c_h = cell_weights["h"]
        else:
            self.c_h = self.create_parameter(
                attr=self._param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in cell_weights and cell_weights["b"] is not None:
            self.c_b = cell_weights["b"]
        else:
            self.c_b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # the weight is concated here in order to make the computation more efficent.
        weight_w = fluid.layers.concat(
            [self.ig_w, self.c_w, self.fg_w, self.og_w], axis=-1)
        weight_h = fluid.layers.concat(
            [self.ig_h, self.c_h, self.fg_h, self.og_h], axis=-1)
        self._weight = fluid.layers.concat([weight_w, weight_h], axis=0)

        self._bias = fluid.layers.concat(
            [self.ig_b, self.c_b, self.fg_b, self.og_b])

    def forward(self, input, state):
        pre_hidden, pre_cell = state
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(
                pre_cell,
                layers.sigmoid(layers.elementwise_add(f, self._forget_bias))),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))
        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, [new_hidden, new_cell]

    @property
    def state_shape(self):
        return [[self._hidden_size], [self._hidden_size]]


class BasicGRUCell(RNNCell):
    """
    ****
    BasicGRUUnit class, using basic operators to build GRU
    The algorithm can be described as the equations below.

        .. math::
            u_t & = actGate(W_ux xu_{t} + W_uh h_{t-1} + b_u)

            r_t & = actGate(W_rx xr_{t} + W_rh h_{t-1} + b_r)

            m_t & = actNode(W_cx xm_t + W_ch dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    Args:
        hidden_size (integer): The hidden size used in the Unit.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            weight matrix. Note:
            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|None): The parameter attribute for the bias
            of GRU unit.
            If it is set to None or one attribute of ParamAttr, gru_unit will 
            create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        gate_activation (function|None): The activation function for gates (actGate).
                                  Default: 'fluid.layers.sigmoid'
        activation (function|None): The activation function for cell (actNode).
                             Default: 'fluid.layers.tanh'
        dtype(string): data type used in this unit
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
        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._dtype = dtype

        assert isinstance(update_gate_weights, dict)
        assert isinstance(reset_gate_weights, dict)
        assert isinstance(cell_weights, dict)

        if self._param_attr is not None and self._param_attr.name is not None:
            gate_param_attr = copy.deepcopy(self._param_attr)
            candidate_param_attr = copy.deepcopy(self._param_attr)
            gate_param_attr.name += "_gate"
            candidate_param_attr.name += "_candidate"
        else:
            gate_param_attr = self._param_attr
            candidate_param_attr = self._param_attr

        if self._bias_attr is not None and self._bias_attr.name is not None:
            gate_bias_attr = copy.deepcopy(self._bias_attr)
            candidate_bias_attr = copy.deepcopy(self._bias_attr)
            gate_bias_attr.name += "_gate"
            candidate_bias_attr.name += "_candidate"
        else:
            gate_bias_attr = self._bias_attr
            candidate_bias_attr = self._bias_attr

        # create the parameters of gates in gru
        if "w" in update_gate_weights and update_gate_weights["w"] is not None:
            self.ug_w = update_gate_weights["w"]
        else:
            self.ug_w = self.create_parameter(
                attr=gate_param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in update_gate_weights and update_gate_weights["h"] is not None:
            self.ug_h = update_gate_weights["h"]
        else:
            self.ug_h = self.create_parameter(
                attr=gate_param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in update_gate_weights and update_gate_weights["b"] is not None:
            self.ug_b = update_gate_weights["b"]
        else:
            self.ug_b = self.create_parameter(
                attr=gate_bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # reset gate parameters
        if "w" in reset_gate_weights and reset_gate_weights["w"] is not None:
            self.rg_w = reset_gate_weights["w"]
        else:
            self.rg_w = self.create_parameter(
                attr=gate_param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in reset_gate_weights and reset_gate_weights["h"] is not None:
            self.rg_h = reset_gate_weights["h"]
        else:
            self.rg_h = self.create_parameter(
                attr=gate_param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in reset_gate_weights and reset_gate_weights["b"] is not None:
            self.rg_b = reused_params["b"]
        else:
            self.rg_b = self.create_parameter(
                attr=gate_param_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        # cell parameters
        if "w" in cell_weights and cell_weights["w"] is not None:
            self.c_w = cell_weights["w"]
        else:
            self.c_w = self.create_parameter(
                attr=candidate_param_attr,
                shape=[self._input_size, self._hidden_size],
                dtype=self._dtype)

        if "h" in cell_weights and cell_weights["h"] is not None:
            self.c_h = cell_weights["h"]
        else:
            self.c_h = self.create_parameter(
                attr=candidate_param_attr,
                shape=[self._hidden_size, self._hidden_size],
                dtype=self._dtype)

        if "b" in cell_weights and cell_weights["b"] is not None:
            self.c_b = cell_weights["b"]
        else:
            self.c_b = self.create_parameter(
                attr=candidate_bias_attr,
                shape=[self._hidden_size],
                dtype=self._dtype,
                is_bias=True)

        rg_weights = layers.concat([self.rg_w, self.rg_h], axis=-1)
        ug_weights = layers.concat([self.ug_w, self.ug_h], axis=-1)
        self._gate_weight = layers.concat([rg_weights, ug_weights], axis=0)

        self._candidate_weight = layers.concat([self.c_w, self.c_h], axis=0)

        self._gate_bias = layers.concat([self.rg_b, self.ug_b], axis=0)

        self._candidate_bias = self.c_b

    def forward(self, input, state):
        pre_hidden = state
        concat_input_hidden = layers.concat([input, pre_hidden], axis=1)

        gate_input = layers.matmul(x=concat_input_hidden, y=self._gate_weight)

        gate_input = layers.elementwise_add(gate_input, self._gate_bias)

        gate_input = self._gate_activation(gate_input)
        r, u = layers.split(gate_input, num_or_sections=2, dim=1)

        r_hidden = r * pre_hidden

        candidate = layers.matmul(
            layers.concat([input, r_hidden], 1), self._candidate_weight)
        candidate = layers.elementwise_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_hidden = u * pre_hidden + (1 - u) * c

        return new_hidden

    @property
    def state_shape(self):
        return [self._hidden_size]


class RNN(fluid.dygraph.Layer):
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

            if sequence_length:
                mask = fluid.layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = fluid.layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=[0]), inputs)
                mask = fluid.layers.reverse(
                    mask, axis=[0]) if sequence_length else None

            states = initial_states
            outputs = []
            for i in range(time_steps):
                step_inputs = map_structure(lambda x: x[i], inputs)
                step_outputs, new_states = self.cell(step_inputs, states,
                                                     **kwargs)
                if sequence_length:
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


class DynamicDecode(Layer):
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
                next_finished = layers.logical_or(next_finished, finished)
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


class TransfomerCell(object):
    """
    Let inputs=(trg_word, trg_pos), states=cache to make Transformer can be
    used as RNNCell
    """

    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, inputs, states, trg_src_attn_bias, enc_output,
                 static_caches):
        trg_word, trg_pos = inputs
        for cache, static_cache in zip(states, static_caches):
            cache.update(static_cache)
        logits = self.decoder(trg_word, trg_pos, None, trg_src_attn_bias,
                              enc_output, states)
        new_states = [{"k": cache["k"], "v": cache["v"]} for cache in states]
        return logits, new_states


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
                self.functors.append(lambda x, y: x + y if y else x)
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
        if attn_bias:
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
