# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import tensorflow as tf
from layers_utils import *
from positional_encoding import *
from transformer_encoder import *
from mha_layer import *
from lmha_layer import *


def bert_encoder(seq_len, dict_len, full_attn, return_intermediate,
                 d_model, dropout_rate, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dim_k, parameter_sharing):
    """
    BERT Transformer-Encoder Model

    Args:
    - seq_len [int]: maximum number of input tokens (maximum length)
    - dict_len [int]: dictionary cardinality (number of unique tokens)
    - full_attn [int]: attention mode: 1 - full attention, 0 - linear attention
    - return_intermediate [int]: 1 - returns the intermediate results
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout
    - num_enc_layers [int]: number of Transformer-Encoder layers
    - num_enc_heads [int]: number of heads of attention
    - enc_dff [int]: number of hidden neurons for the first dense layer of the PWFFN
    - enc_atv_fun: PWFFN dense layers activation function
    - dim_k [int]: Linear MHA projection dimension
    - parameter_sharing [str]: Linear MHA parameter sharing option

    """

    # Input Layer
    inputs = tf.keras.Input(shape=seq_len, dtype=tf.int64, name='input_layer')

    # Padding Masking Matrix
    pad_mask = attn_pad_mask()(inputs)

    # Token Embedding Layer
    input_embedding = tf.keras.layers.Embedding(dict_len, d_model, name='emb_layer')(inputs)

    # Positional Embedding Layer
    char_embedding = pos_emb_layer(seq_len, d_model, dropout_rate, name='pos_layer')(input_embedding)

    # Transformer-Encoder Block: MHSA Layer + PWFFN
    trans_encoder, attn_weights = transformer_encoder(d_model, num_enc_layers, num_enc_heads, enc_dff,
                                          enc_atv_fun, dropout_rate,
                                          dim_k, parameter_sharing, full_attn, return_intermediate,
                                          name='trans_encoder')(char_embedding, pad_mask)

    return tf.keras.Model(inputs=inputs, outputs=[trans_encoder, attn_weights])
