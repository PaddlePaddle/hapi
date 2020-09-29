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

import paddle
from paddle.nn import Layer, Linear, Dropout, Embedding, LayerList, RNN, LSTM, LSTMCell, RNNCellBase
from paddle.fluid.layers import BeamSearchDecoder, dynamic_decode
import paddle.nn.functional as F
import paddle.nn.initializer as I


class CrossEntropyCriterion(Layer):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, predict, trg_mask, label):
        cost = F.softmax_with_cross_entropy(
            logits=predict, label=label, soft_label=False)
        cost = paddle.squeeze(cost, axis=[2])
        masked_cost = cost * trg_mask
        batch_mean_cost = paddle.reduce_mean(masked_cost, dim=[0])
        seq_cost = paddle.reduce_sum(batch_mean_cost)
        return seq_cost


class Encoder(Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 padding_idx=0,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Encoder, self).__init__()
        self.embedder = Embedding(
            vocab_size,
            embed_dim,
            padding_idx,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction="forward",
            dropout=dropout_prob if num_layers > 1 else 0., )

    def forward(self, sequence, sequence_length):
        inputs = self.embedder(sequence)
        encoder_output, encoder_state = self.lstm(
            inputs, sequence_length=sequence_length)

        return encoder_output, encoder_state


class AttentionLayer(Layer):
    def __init__(self, hidden_size, bias=False, init_scale=0.1):
        super(AttentionLayer, self).__init__()
        self.input_proj = Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)
        self.output_proj = Linear(
            hidden_size + hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=bias)

    def forward(self, hidden, encoder_output, encoder_padding_mask):
        query = self.input_proj(hidden)
        attn_scores = paddle.matmul(
            paddle.unsqueeze(query, [1]), encoder_output, transpose_y=True)
        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)
        attn_scores = F.softmax(attn_scores)
        attn_out = paddle.squeeze(
            paddle.matmul(attn_scores, encoder_output), [1])
        attn_out = paddle.concat([attn_out, hidden], 1)
        attn_out = self.output_proj(attn_out)
        return attn_out


class DecoderCell(RNNCellBase):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(DecoderCell, self).__init__()
        if dropout_prob > 0:
            self.dropout = Dropout(dropout_prob)
        else:
            self.dropout = None

        self.lstm_cells = LayerList([
            LSTMCell(
                input_size=input_size + hidden_size if i == 0 else hidden_size,
                hidden_size=hidden_size) for i in range(num_layers)
        ])
        self.attention_layer = AttentionLayer(hidden_size)

    def forward(self,
                step_input,
                states,
                encoder_output=None,
                encoder_padding_mask=None,
                attention=True):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = paddle.concat([step_input, input_feed], 1)
        for i, lstm_cell in enumerate(self.lstm_cells):
            out, new_lstm_state = lstm_cell(step_input, lstm_states[i])
            if self.dropout:
                out = self.dropout(out)
            step_input = out
            new_lstm_states.append(new_lstm_state)
        if attention:
            out = self.attention_layer(step_input, encoder_output,
                                       encoder_padding_mask)
        else:
            out = step_input
        return out, [new_lstm_states, out]


class Decoder(Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 padding_idx=0,
                 attention=True,
                 dropout_prob=0.,
                 init_scale=0.1):
        super(Decoder, self).__init__()
        self.attention = attention
        self.embedder = Embedding(
            vocab_size,
            embed_dim,
            padding_idx,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        self.lstm_attention = RNN(DecoderCell(
            num_layers, embed_dim, hidden_size, dropout_prob, init_scale),
                                  is_reverse=False,
                                  time_major=False)
        self.output_layer = Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=False)

    def forward(self,
                target,
                decoder_initial_states,
                encoder_output=None,
                encoder_padding_mask=None):
        inputs = self.embedder(target)
        decoder_output, _ = self.lstm_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            attention=self.attention)
        predict = self.output_layer(decoder_output)
        return predict


class Seq2Seq(Layer):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 attention=True,
                 dropout_prob=0.,
                 padding_idx=0,
                 init_scale=0.1):
        super(Seq2Seq, self).__init__()
        self.attention = attention
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_size,
                               num_layers, padding_idx, dropout_prob,
                               init_scale)
        self.decoder = Decoder(trg_vocab_size, embed_dim, hidden_size,
                               num_layers, padding_idx, attention,
                               dropout_prob, init_scale)

    def forward(self, src, src_length, trg):
        # encoder
        encoder_output, encoder_final_states = self.encoder(src, src_length)

        # decoder initial states
        decoder_initial_states = [
            encoder_final_states,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        if self.attention:
            # attention mask to avoid paying attention on padddings
            src_mask = (
                src != self.padding_idx).astype(paddle.get_default_dtype())
            encoder_padding_mask = (src_mask - 1.0) * 1e9
            encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])
            # decoder with attentioon
            predict = self.decoder(trg, decoder_initial_states, encoder_output,
                                   encoder_padding_mask)
        else:
            predict = self.decoder(trg, decoder_initial_states)

        trg_mask = (trg != self.padding_idx).astype(paddle.get_default_dtype())
        return predict, trg_mask


class Seq2SeqInfer(Seq2Seq):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 attention=True,
                 dropout_prob=0.,
                 padding_idx=0,
                 bos_id=0,
                 eos_id=1,
                 beam_size=4,
                 max_out_len=256):
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)  # py3
        self.bos_id = args.pop("bos_id")
        self.eos_id = args.pop("eos_id")
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        super(Seq2SeqInfer, self).__init__(**args)
        # dynamic decoder for inference
        decoder = BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedder,
            output_fn=self.decoder.output_layer)

    def forward(self, src, src_length):
        # encoding
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        # decoder initial states
        decoder_initial_states = [
            encoder_final_state,
            self.decoder.lstm_attention.cell.get_initial_states(
                batch_ref=encoder_output, shape=[self.hidden_size])
        ]
        rs = dynamic_decode(decoder=decoder, inits=decoder_initial_states)
        return rs
