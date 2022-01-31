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

from paddle.io import Dataset
"""data generator"""
class DataGenerator(Dataset):
    def __init__(self, features, args):
        super(DataGenerator, self).__init__()
        self.args = args
        self.all_guid = [f.guid for f in features]
        self.all_text_a = [f.text_a for f in features]
        self.all_text_b = [f.text_b[1:] for f in features]
        self.all_label = [[f.label] for f in features]

    def __getitem__(self, item):
        text_a = self.all_text_a[item]
        text_b = self.all_text_b[item]
        text_a_attention_mask = [1] * len(text_a)
        text_b_attention_mask = [1] * len(text_b)
        text_a_token_type_ids = [0] * len(text_a)
        text_b_token_type_ids = [1] * len(text_b)
        label = self.all_label[item]

        return text_a, text_b, text_a_attention_mask, \
               text_b_attention_mask, text_a_token_type_ids, \
               text_b_token_type_ids, label

    def __len__(self):
        return len(self.all_text_a)