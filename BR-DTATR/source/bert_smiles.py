# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import tensorflow as tf
from transformer_encoder import *
from embedding_layer import *
import time
import os
import itertools
import tensorflow_addons as tfa
from layers_utils import *
from positional_encoding import *


def bert_smiles_mlm(seq_len, dict_len, full_attn, return_intermediate,
             d_model, dropout_rate, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dim_k, parameter_sharing):

    """
    Function to build the SMILES Pre-Train MLM Model
    Args:
    - seq_len [int]: maximum number of input tokens (length)
    - dict_len [int]: dictionary cardinality (number of unique tokens)
    - full_attn [int]: attention mode: 1 - full attention, 0 - linear attention
    - return_intermediate [int]: 1 - returns the intermediate results
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout
    - num_enc_layers [int]: number of SMILES Transformer-Encoders
    - num_enc_heads [int]: number of heads for the SMILES MHSA
    - enc_dff [int]: hidden numbers for the first dense layer of the PWFFN
    - enc_atv_fun: dense layers activation function
    - dim_k [int]: projection value (in the case of Linear MHSA)
    - parameter_sharing [str]: parameter sharing option in the case of Linear MHSA

    Outputs:
    - SMILES Pre-Train MLM Model
    """


    # if enc_atv_fun == 'gelu':
    #     enc_atv_fun = gelu

    # Input Layer
    inputs = tf.keras.Input(shape=seq_len+1, dtype=tf.int64, name='input_layer')

    # Padding Masking Matrix
    pad_mask = attn_pad_mask()(inputs)

    # Token Embedding Layer
    input_embedding = tf.keras.layers.Embedding(dict_len + 3, d_model, name='emb_layer')(inputs)

    # Positional Embedding Layer
    char_embedding = pos_emb_layer(seq_len + 1, d_model, dropout_rate, name='pos_layer')(input_embedding)

    # Transformer-Encoder Block: MHSA Layer + PWFFN
    trans_encoder, _ = transformer_encoder(d_model, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dropout_rate, dim_k, parameter_sharing,
                             full_attn, return_intermediate, name='trans_encoder')(char_embedding, pad_mask)


    # Output Dense Layer: Projects the last dimension into the dimension of the dictionary vocabulary
    outputs = tf.keras.layers.Dense(units=dict_len+3, name='mlm_cls')(trans_encoder)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# def bert_smiles(model):
#     """
#     Function to transform the SMILES Pre-Train MLM Architecture into the SMILES Transformer-Encoder Architecture
#     Args:
#     - model: SMILES Pre-Train MLM Model

#     Outputs:
#     - Pre-Trained SMILES Transformer-Encoder
#     """

#     new_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('trans_encoder').output)
#     new_model._name = 'smiles_bert'
#     return new_model


# Load Saved Pre-trained SMILES MLM Model & Return SMILES Transformer-Encoder Architecture
def load_bert_smiles(model, checkpoint_path):
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-03, beta_1=0.9,
                                               beta_2=0.999, epsilon=1e-08,
                                               weight_decay=1e-04, total_steps=512500,
                                               warmup_proportion=0.01, min_lr=1e-05)
                                               
    ckpt_obj = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)

    # Restore Saved Pre-trained Model
    latest = tf.train.latest_checkpoint(checkpoint_path)
    ckpt_obj.restore(latest).expect_partial()
    model = ckpt_obj.model

    # Transform the SMILES Pre-Train MLM Architecture into the SMILES Transformer-Encoder Architecture
    new_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('trans_encoder').output)
    new_model._name = 'smiles_bert'

    return new_model





















