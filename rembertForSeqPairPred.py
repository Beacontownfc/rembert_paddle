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

from rembert.rembert_model import RembertPretrainedModel
import paddle.nn as nn

class RembertForSeqPairPred(RembertPretrainedModel):
    def __init__(self, rembert, num_label):
        super().__init__()
        self.rembert = rembert
        self.dense = nn.Linear(self.rembert.config['hidden_size'], num_label)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        pool_output = self.rembert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)[1]

        pool_output = self.dropout(pool_output)
        logits = self.dense(pool_output)
        if labels is not None:
            loss = self.loss_fn(logits, labels.reshape([-1]))
            return loss, logits
        else:
            return logits



    