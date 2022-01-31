#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import argparse

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from tqdm import tqdm
from trainer import Trainer
import pickle
import numpy as np
import copy
from rembertForSeqPairPred import RembertForSeqPairPred
from dataProcessor import MrpcProcessor, tokenization, XNLIProcessor
from datagenerator import DataGenerator


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default='/home/aistudio/data/data126002/')
parser.add_argument("--do_eval", type=int, default=0)
parser.add_argument("--do_train", type=int, default=0)
parser.add_argument("--eval_batch_size", type=int, default=24)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--pretrain_model", default='/home/aistudio/data/data125938/')
parser.add_argument("--output_dir", default="/home/aistudio/output/")
parser.add_argument("--max_seq_length", default=512)
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--gradient_accumulation_steps", default=3)
parser.add_argument("--warmup_proportion", type=float, default=0.06)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--adam_b1", type=float, default=0.9)
parser.add_argument("--adam_b2", type=float, default=0.999)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--task", type=str, required=True)
args = parser.parse_args()

def check_state_dict(cur_model_state_dict, trg_model_state_dict):
    """Check whether the parameter are loaded correctly"""
    def mul(shape):
        s = 1
        for e in shape:
            s *= e
        return s

    for k, v in cur_model_state_dict.items():
        x = trg_model_state_dict.get(k, False)
        if isinstance(x, bool):
            print(k + " 参数不存在!!!")
        else:
            if x.shape != v.shape or (v == x).sum() != mul(v.shape):
                print(k + ' 参数不正确!!!')


def load_example(args, fold='train'):
    """Load data to DataLoader"""
    if args.task == 'paws':
        processor = MrpcProcessor()
    if args.task == 'xnli':
        processor = XNLIProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    datagenerator = DataGenerator(examples, args)

    def collate_fn(batch):
        """Pad sequence"""
        def create_padded_sequence(k, padding_value):
            new_data = []
            max_len = 0
            for each_batch in batch:
                if len(each_batch[k]) > max_len:
                    max_len = len(each_batch[k])
            for each_batch in batch:
                new_data.append(each_batch[k] + [padding_value] * (max_len - len(each_batch[k])))
            return np.array(new_data, dtype='int64')

        text_a = create_padded_sequence(0, tokenization.pad_token_id)
        text_b = create_padded_sequence(1, tokenization.pad_token_id)
        text_a_attention_mask = create_padded_sequence(2, 0)
        text_b_attention_mask = create_padded_sequence(3, 0)
        text_a_token_type_ids = create_padded_sequence(4, 0)
        text_b_token_type_ids = create_padded_sequence(5, 1)
        label = create_padded_sequence(6, 0)

        input_ids = np.concatenate([text_a, text_b], axis=-1)[:, :args.max_seq_length]
        attention_mask = np.concatenate([text_a_attention_mask, text_b_attention_mask], axis=-1)[:, :args.max_seq_length]
        token_type_ids = np.concatenate([text_a_token_type_ids, text_b_token_type_ids], axis=-1)[:, :args.max_seq_length]

        return (input_ids, attention_mask, token_type_ids, label)
    
    #nranks = paddle.distributed.ParallelEnv().nranks

    if fold in ("dev", "test"):
        dataloader = DataLoader(datagenerator, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(datagenerator, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)
            
    return dataloader, processor

def run(args):
    if args.do_train:
        if args.task == 'paws':
            num_label = 2
        if args.task == 'xnli':
            num_label = 3
        model = RembertForSeqPairPred.from_pretrained(args.pretrain_model, num_label=num_label)
        train_dataloader, processor = load_example(args, 'train')
        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)
        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
        trainer.train()
    
    if args.do_eval:
        if args.task == 'paws':
            num_label = 2
        if args.task == 'xnli':
            num_label = 3
        
        state_dict= paddle.load(args.output_dir + 'model_state.pdparams')
        for k, v in state_dict.items():
            if v.dtype == paddle.float16:
                v = v.astype('float32')
            state_dict[k] = v
        paddle.save(state_dict, args.output_dir + 'model_state.pdparams')
        del state_dict
        
        model = RembertForSeqPairPred.from_pretrained(args.output_dir, num_label=num_label)
        evaluate(model, args)


def evaluate(model, args):
    """evaluate the model"""
    model.eval()
    total_corr = 0
    total_pred = 0
    eval_dataloader, processor = load_example(args, 'test')
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        logits = model(input_ids=batch[0],
                       attention_mask=batch[1],
                       token_type_ids=batch[2])
        pred = logits.astype('float32').argmax(-1)
        labels = batch[3].reshape([-1])
        total_corr += (pred == labels).astype('float32').sum().tolist()[0]
        total_pred += len(batch[3])
    print('Acc:', total_corr / total_pred)
        

if __name__ == '__main__':
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    run(args)


