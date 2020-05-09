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
Emotion Detection Task in Paddle Dygraph Mode. 
"""

from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy as np
from hapi.model import set_device, CrossEntropy, Input
from hapi.metrics import Accuracy
from hapi.text.emo_tect import EmoTectProcessor
from models import CNN, BOW, GRU, BiGRU, LSTM
from hapi.configure import Config
import json

def main():
    """
    Main Function
    """
    args = Config(yaml_file='./config.yaml')
    args.build()
    args.Print()
    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
                         "least one of them must be True.")

    place = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(place)

    processor = EmoTectProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        random_seed=args.random_seed)
    num_labels = args.num_labels

    if args.model_type == 'cnn_net':
        model = CNN( args.vocab_size, 
                     args.max_seq_len)
    elif args.model_type == 'bow_net':
        model = BOW( args.vocab_size, 
                     args.max_seq_len)
    elif args.model_type == 'lstm_net':
        model = LSTM( args.vocab_size, 
                     args.max_seq_len)
    elif args.model_type == 'gru_net':
        model = GRU( args.vocab_size, 
                     args.max_seq_len)
    elif args.model_type == 'bigru_net':
        model = BiGRU( args.vocab_size, 
                       args.batch_size,
                       args.max_seq_len)
    else:
        raise ValueError("Unknown model type!")

    inputs = [Input([None, args.max_seq_len], 'int64', name='doc')]
    optimizer = None
    labels = None

    if args.do_train:
        train_data_generator = processor.data_generator(
            batch_size=args.batch_size, 
            places=place,
            phase='train', 
            epoch=args.epoch, 
            padding_size=args.max_seq_len)

        num_train_examples = processor.get_num_examples(phase="train")
        max_train_steps = args.epoch * num_train_examples // args.batch_size + 1

        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        labels = [Input([None, 1], 'int64', name='label')]
        optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr, parameter_list=model.parameters())
        test_data_generator = None
        if args.do_val:
            test_data_generator = processor.data_generator(
                batch_size=args.batch_size, 
                phase='dev', 
                epoch=1, 
                places=place,
                padding_size=args.max_seq_len)

    elif args.do_val:
        test_data_generator = processor.data_generator(
            batch_size=args.batch_size, 
            phase='test', 
            epoch=1,
            places=place,
            padding_size=args.max_seq_len)

    elif args.do_infer:
        infer_data_generator = processor.data_generator(
            batch_size=args.batch_size, 
            phase='infer', 
            epoch=1,
            places=place,
            padding_size=args.max_seq_len)

    model.prepare(
    optimizer,
    CrossEntropy(),
    Accuracy(topk=(1,)),
    inputs,
    labels,
    device=place)

    if args.do_train:
        if args.init_checkpoint:
            model.load(args.init_checkpoint)
    elif args.do_val or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or infer!")
        model.load(args.init_checkpoint, reset_optimizer=True)

    if args.do_train:
        model.fit(train_data=train_data_generator,
                  eval_data=test_data_generator,
                  batch_size=args.batch_size,
                  epochs=args.epoch,
                  save_dir=args.checkpoints,
                  eval_freq=args.eval_freq,
                  save_freq=args.save_freq)
    elif args.do_val:
        eval_result = model.evaluate(eval_data=test_data_generator,
                                     batch_size=args.batch_size)
        print("Final eval result: acc: {:.4f}, loss: {:.4f}".format(eval_result['acc'], eval_result['loss'][0]))

    elif args.do_infer:
        preds = model.predict(test_data=infer_data_generator)
        preds = np.array(preds[0]).reshape((-1, args.num_labels))

        if args.output_dir:
            with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as w:

                for p in range(len(preds)):
                    label = np.argmax(preds[p])
                    result = json.dumps({'index': p, 'label': label, 'probs': preds[p].tolist()})
                    w.write(result+'\n')
            print('Predictions saved at '+os.path.join(args.output_dir, 'predictions.json'))

if __name__ == "__main__":
    main()
