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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import BeamSearchDecoder, dynamic_decode


class ConvBNPool(paddle.nn.Layer):
    def __init__(self,
                 in_ch,
                 out_ch,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
        super(ConvBNPool, self).__init__()
        self.pool = pool

        filter_size = 3
        std = (2.0 / (filter_size**2 * in_ch))**0.5
        param_0 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, std))

        std = (2.0 / (filter_size**2 * out_ch))**0.5
        param_1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, std))

        net = [
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
                weight_attr=param_0,
                bias_attr=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act == 'relu':
            net += [nn.ReLU()]

        net += [
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                weight_attr=param_1,
                bias_attr=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act == 'relu':
            net += [nn.ReLU()]

        if self.pool:
            net += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return self.net(inputs)


class CNN(paddle.nn.Layer):
    def __init__(self, in_ch=1, is_test=False):
        super(CNN, self).__init__()
        net = [
            ConvBNPool(in_ch, 16),
            ConvBNPool(16, 32),
            ConvBNPool(32, 64),
            ConvBNPool(
                64, 128, pool=False),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return self.net(inputs)


class Encoder(paddle.nn.Layer):
    def __init__(
            self,
            in_channel=1,
            rnn_hidden_size=200,
            decoder_size=128,
            is_test=False, ):
        super(Encoder, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size

        self.backbone = CNN(in_ch=in_channel, is_test=is_test)

        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, 0.02),
            learning_rate=2.0)
        self.gru_fwd = nn.RNN(cell=nn.GRUCell(
            input_size=128 * 6, hidden_size=rnn_hidden_size),
                              is_reverse=False,
                              time_major=False)
        self.gru_bwd = nn.RNN(cell=nn.GRUCell(
            input_size=128 * 6, hidden_size=rnn_hidden_size),
                              is_reverse=True,
                              time_major=False)
        self.encoded_proj_fc = nn.Linear(
            rnn_hidden_size * 2, decoder_size, bias_attr=False)

    def forward(self, inputs):
        conv_features = self.backbone(inputs)
        conv_features = paddle.transpose(conv_features, perm=[0, 3, 1, 2])

        n, w, c, h = conv_features.shape
        seq_feature = paddle.reshape(conv_features, [0, -1, c * h])

        gru_fwd, _ = self.gru_fwd(seq_feature)
        gru_bwd, _ = self.gru_bwd(seq_feature)

        encoded_vector = paddle.concat([gru_fwd, gru_bwd], axis=2)
        encoded_proj = self.encoded_proj_fc(encoded_vector)
        return gru_bwd, encoded_vector, encoded_proj


class Attention(paddle.nn.Layer):
    """
    Neural Machine Translation by Jointly Learning to Align and Translate.
    https://arxiv.org/abs/1409.0473
    """

    def __init__(self, decoder_size):
        super(Attention, self).__init__()
        self.fc1 = nn.Linear(decoder_size, decoder_size, bias_attr=False)
        self.fc2 = nn.Linear(decoder_size, 1, bias_attr=False)

    def forward(self, encoder_vec, encoder_proj, decoder_state):
        # alignment model, single-layer multilayer perceptron
        decoder_state = self.fc1(decoder_state)
        decoder_state = paddle.unsqueeze(decoder_state, [1])

        e = paddle.add(encoder_proj, decoder_state)
        e = paddle.tanh(e)

        att_scores = self.fc2(e)
        att_scores = paddle.squeeze(att_scores, [2])
        att_scores = F.softmax(att_scores)

        context = paddle.multiply(encoder_vec, att_scores, axis=0)
        context = paddle.sum(context, axis=1)
        return context


class DecoderCell(nn.RNNCellBase):
    def __init__(self, encoder_size=200, decoder_size=128):
        super(DecoderCell, self).__init__()
        self.attention = Attention(decoder_size)
        self.gru_cell = nn.GRUCell(
            input_size=encoder_size * 2 + decoder_size,
            hidden_size=decoder_size)

    def forward(self, current_word, states, encoder_vec, encoder_proj):
        context = self.attention(encoder_vec, encoder_proj, states)
        decoder_inputs = paddle.concat([current_word, context], axis=1)
        hidden, _ = self.gru_cell(decoder_inputs, states)
        return hidden, hidden


class Decoder(paddle.nn.Layer):
    def __init__(self, num_classes, emb_dim, encoder_size, decoder_size):
        super(Decoder, self).__init__()
        self.decoder_attention = nn.RNN(
            DecoderCell(encoder_size, decoder_size))
        self.fc = nn.Linear(decoder_size, num_classes + 2)

    def forward(self, target, initial_states, encoder_vec, encoder_proj):
        out, _ = self.decoder_attention(
            target,
            initial_states=initial_states,
            encoder_vec=encoder_vec,
            encoder_proj=encoder_proj)
        pred = self.fc(out)
        return pred


class Seq2SeqAttModel(paddle.nn.Layer):
    def __init__(
            self,
            in_channle=1,
            encoder_size=200,
            decoder_size=128,
            emb_dim=128,
            num_classes=None, ):
        super(Seq2SeqAttModel, self).__init__()
        self.encoder = Encoder(in_channle, encoder_size, decoder_size)
        self.fc = nn.Sequential(
            nn.Linear(
                encoder_size, decoder_size, bias_attr=False), nn.ReLU())
        self.embedding = nn.Embedding(num_classes + 2, emb_dim)
        self.decoder = Decoder(num_classes, emb_dim, encoder_size,
                               decoder_size)

    def forward(self, inputs, target):
        gru_backward, encoded_vector, encoded_proj = self.encoder(inputs)
        decoder_boot = self.fc(gru_backward[:, 0])
        trg_embedding = self.embedding(target)
        prediction = self.decoder(trg_embedding, decoder_boot, encoded_vector,
                                  encoded_proj)
        return prediction


class Seq2SeqAttInferModel(Seq2SeqAttModel):
    def __init__(
            self,
            in_channle=1,
            encoder_size=200,
            decoder_size=128,
            emb_dim=128,
            num_classes=None,
            beam_size=0,
            bos_id=0,
            eos_id=1,
            max_out_len=20, ):
        super(Seq2SeqAttInferModel, self).__init__(
            in_channle, encoder_size, decoder_size, emb_dim, num_classes)
        self.beam_size = beam_size
        # dynamic decoder for inference
        decoder = BeamSearchDecoder(
            self.decoder.decoder_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.embedding,
            output_fn=self.decoder.fc)
        self.max_out_len == max_out_len

    def forward(self, inputs, *args):
        gru_backward, encoded_vector, encoded_proj = self.encoder(inputs)
        decoder_boot = self.fc(gru_backward[:, 0])

        if self.beam_size:
            # Tile the batch dimension with beam_size
            encoded_vector = BeamSearchDecoder.tile_beam_merge_with_batch(
                encoded_vector, self.beam_size)
            encoded_proj = BeamSearchDecoder.tile_beam_merge_with_batch(
                encoded_proj, self.beam_size)
        # dynamic decoding with beam search
        rs, _ = dynamic_decode(
            inits=decoder_boot,
            max_step_num=self.max_out_len,
            is_test=True,
            encoder_vec=encoded_vector,
            encoder_proj=encoded_proj)
        return rs


class WeightCrossEntropy(paddle.nn.Layer):
    def __init__(self):
        super(WeightCrossEntropy, self).__init__()

    def forward(self, predict, label, mask):
        predict = paddle.flatten(predict, start_axis=0, stop_axis=1)
        label = paddle.reshape(label, shape=[-1, 1])
        mask = paddle.reshape(mask, shape=[-1, 1])
        loss = F.cross_entropy(predict, label=label)
        loss = paddle.multiply(loss, mask, axis=0)
        loss = paddle.sum(loss)
        return loss
