# encoding=utf8
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

from paddlenlp.transformers import PretrainedModel, register_base_model
import paddle
import paddle.nn as nn
import math


class RembertPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "rembert": {"attention_probs_dropout_prob": 0,
                    "bos_token_id": 312,
                    "embedding_size": 256,
                    "eos_token_id": 313,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0,
                    "hidden_size": 1152,
                    "initializer_range": 0.02,
                    "intermediate_size": 4608,
                    "layer_norm_eps": 1e-12,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 18,
                    "num_hidden_layers": 32,
                    "pad_token_id": 0,
                    "type_vocab_size": 2,
                    "vocab_size": 250300,
                    "init_class": "RemBertModel"}
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "rembert":
                "https://bj.bcebos.com/paddlenlp/models/transformers/rembert/model_state.pdparams",
        }
    }
    base_model_prefix = "rembert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.rembert.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = self.layer_norm_eps if hasattr(
                self, "layer_norm_eps") else self.rembert.config["layer_norm_eps"]


class RemBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self,
                 pad_token_id=0,
                 vocab_size=250300,
                 input_embedding_size=256,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0
                 ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, input_embedding_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, input_embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, input_embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(input_embedding_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(end=max_position_embeddings).expand((1, -1)))

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->RemBert
class RemBertPooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RemBertSelfAttention(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 num_attention_heads=18,
                 attention_probs_dropout_prob=0,
                 is_decoder=False
                 ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.is_decoder = is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RemBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RemBert
class RemBertSelfOutput(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->RemBert
class RemBertAttention(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 num_attention_heads=18,
                 attention_probs_dropout_prob=0,
                 hidden_dropout_prob=0,
                 layer_norm_eps=1e-12,
                 is_decoder=False):
        super().__init__()
        self.self = RemBertSelfAttention(hidden_size=hidden_size,
                                         num_attention_heads=num_attention_heads,
                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                         is_decoder=is_decoder)
        self.output = RemBertSelfOutput(hidden_size=hidden_size,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        layer_norm_eps=layer_norm_eps
                                        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RemBert
class RemBertIntermediate(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 intermediate_size=4608
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RemBert
class RemBertOutput(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 hidden_dropout_prob=0,
                 intermediate_size=4608,
                 layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->RemBert
class RemBertLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1152,
                 num_attention_heads=18,
                 attention_probs_dropout_prob=0,
                 hidden_dropout_prob=0,
                 layer_norm_eps=1e-12,
                 intermediate_size=4608,
                 is_decoder=False,
                 add_cross_attention=None,
                 ):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = RemBertAttention(hidden_size=hidden_size,
                                          num_attention_heads=num_attention_heads,
                                          attention_probs_dropout_prob=attention_probs_dropout_prob,
                                          hidden_dropout_prob=hidden_dropout_prob,
                                          layer_norm_eps=layer_norm_eps,
                                          is_decoder=is_decoder)
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder "
                                 f"model if cross attention is added")
            self.crossattention = RemBertAttention(hidden_size=hidden_size,
                                                   num_attention_heads=num_attention_heads,
                                                   attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                   hidden_dropout_prob=hidden_dropout_prob,
                                                   layer_norm_eps=layer_norm_eps,
                                                   is_decoder=is_decoder)
        self.intermediate = RemBertIntermediate(hidden_size=hidden_size,
                                                intermediate_size=intermediate_size)
        self.output = RemBertOutput(hidden_size=hidden_size,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    intermediate_size=intermediate_size,
                                    layer_norm_eps=layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} "
                    f"has to be instantiated with cross-attention layers "
                    f"by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RemBertEncoder(nn.Layer):
    def __init__(self,
                 input_embedding_size=256,
                 hidden_size=1152,
                 num_hidden_layers=32,
                 num_attention_heads=18,
                 attention_probs_dropout_prob=0,
                 hidden_dropout_prob=0,
                 layer_norm_eps=1e-12,
                 intermediate_size=4608,
                 is_decoder=False,
                 add_cross_attention=None,
                 ):
        super().__init__()
        self.add_cross_attention = add_cross_attention
        self.embedding_hidden_mapping_in = nn.Linear(input_embedding_size, hidden_size)
        self.layer = nn.LayerList([RemBertLayer(hidden_size=hidden_size,
                                                num_attention_heads=num_attention_heads,
                                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                hidden_dropout_prob=hidden_dropout_prob,
                                                layer_norm_eps=layer_norm_eps,
                                                intermediate_size=intermediate_size,
                                                is_decoder=is_decoder,
                                                add_cross_attention=add_cross_attention) for _ in
                                   range(num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        ]
                     if v is not None)


@register_base_model
class RemBertModel(RembertPretrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self,
                 vocab_size=250300,
                 input_embedding_size=256,
                 type_vocab_size=2,
                 max_position_embeddings=512,
                 hidden_size=1152,
                 num_hidden_layers=32,
                 num_attention_heads=18,
                 attention_probs_dropout_prob=0,
                 hidden_dropout_prob=0,
                 layer_norm_eps=1e-12,
                 initializer_range=0.02,
                 intermediate_size=4608,
                 pad_token_id=0,
                 is_decoder=False,
                 add_cross_attention=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 add_pooling_layer=True):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.is_decoder = is_decoder
        self.embeddings = RemBertEmbeddings(vocab_size=vocab_size,
                                            pad_token_id=pad_token_id,
                                            input_embedding_size=input_embedding_size,
                                            max_position_embeddings=max_position_embeddings,
                                            type_vocab_size=type_vocab_size,
                                            layer_norm_eps=layer_norm_eps,
                                            hidden_dropout_prob=hidden_dropout_prob)
        self.encoder = RemBertEncoder(input_embedding_size=input_embedding_size,
                                      hidden_size=hidden_size,
                                      num_hidden_layers=num_hidden_layers,
                                      num_attention_heads=num_attention_heads,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      layer_norm_eps=layer_norm_eps,
                                      intermediate_size=intermediate_size,
                                      is_decoder=is_decoder,
                                      add_cross_attention=add_cross_attention)

        self.pooler = RemBertPooler(hidden_size=hidden_size) if add_pooling_layer else None

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        if self.is_decoder:
            use_cache = use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length + past_key_values_length))
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif len(attention_mask.shape) == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            if self.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = paddle.arange(end=seq_length)
                causal_mask = seq_ids[None, None, :].repeat((batch_size,
                                                            seq_length, 1)) <= seq_ids[None, :, None]

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = paddle.concat(
                        [
                            paddle.ones(
                                (batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or "
                f"attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if len(encoder_attention_mask.shape) == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if len(encoder_attention_mask.shape) == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))

        if encoder_extended_attention_mask.dtype == paddle.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif encoder_extended_attention_mask.dtype == paddle.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9

        return encoder_extended_attention_mask
    
    def get_head_mask(self, 
                      head_mask, 
                      num_hidden_layers, 
                      is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if len(head_mask.shape) == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif len(head_mask.shape) == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert len(head_mask.shape) == 5, f"head_mask.dim != 5, instead {len(head_mask.shape)}"
        return head_mask


