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

from hapi.text.text import RNNCell as RNNCell
from hapi.text.text import BasicLSTMCell as BasicLSTMCell
from hapi.text.text import BasicGRUCell as BasicGRUCell
from hapi.text.text import RNN as RNN
from hapi.text.text import StackedLSTMCell as StackedLSTMCell
from hapi.text.text import LSTM as LSTM
from hapi.text.text import BidirectionalLSTM as BidirectionalLSTM
from hapi.text.text import StackedGRUCell as StackedGRUCell
from hapi.text.text import GRU as GRU
from hapi.text.text import BidirectionalGRU as BidirectionalGRU
from hapi.text.text import DynamicDecode as DynamicDecode
from hapi.text.text import BeamSearchDecoder as BeamSearchDecoder

from hapi.text.text import Conv1dPoolLayer as Conv1dPoolLayer
from hapi.text.text import CNNEncoder as CNNEncoder

from hapi.text.text import MultiHeadAttention as MultiHeadAttention
from hapi.text.text import FFN as FFN
from hapi.text.text import TransformerEncoderLayer as TransformerEncoderLayer
from hapi.text.text import TransformerDecoderLayer as TransformerDecoderLayer
from hapi.text.text import TransformerEncoder as TransformerEncoder
from hapi.text.text import TransformerDecoder as TransformerDecoder
from hapi.text.text import TransformerCell as TransformerCell
from hapi.text.text import TransformerBeamSearchDecoder as TransformerBeamSearchDecoder

from hapi.text.text import GRUCell as GRUCell
from hapi.text.text import GRUEncoderCell as GRUEncoderCell
from hapi.text.text import BiGRU as BiGRU
from hapi.text.text import LinearChainCRF as LinearChainCRF
from hapi.text.text import CRFDecoding as CRFDecoding
from hapi.text.text import SequenceTagging as SequenceTagging
