# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import math


# GELU activation function
def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class pwffn_block(tf.keras.layers.Layer):
    """
    Feed-Forward Network (FFN): Position-Wise (Dense layers applied to the last dimension)
    - The first dense layer initially projects the last dimension of the input to
    a higher dimension with a certain expansion ratio
    - The second dense layer projects it back to the initial last dimension

    Args:
    - d_model [int]: embedding dimension
    - d_ff [int]: number of hidden neurons for the first dense layer (expansion ratio)
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout

    """

    def __init__(self, d_model, d_ff, atv_fun, dropout_rate, **kwargs):
        super(pwffn_block, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(units=self.d_ff, activation=self.atv_fun)
        self.dense_2 = tf.keras.layers.Dense(units=self.d_model, activation=self.atv_fun)
        self.dropout_layer_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        """

        Args:
        - x: attention outputs

        Shape:
        - Inputs:
        - x: (B,L,E) where B is the batch size, L is the sequence length, E is the embedding dimension
        - Outputs:
        - x: (B,L,E) where B is the batch size, L is the input sequence length, E is the embedding dimension

        """

        x = self.dense_1(x)
        x = self.dropout_layer_1(x)
        x = self.dense_2(x)
        x = self.dropout_layer_2(x)

        return x

    def get_config(self):
        config = super(pwffn_block, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate})
        return config


class attn_pad_mask(tf.keras.layers.Layer):
    """
    Attention Padding Mask Layer: Creates the Padding mask for the attention weights

    """

    def __init__(self, **kwargs):
        super(attn_pad_mask, self).__init__(**kwargs)

        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x, 0), dtype=tf.float32))
        self.reshape_layer = tf.keras.layers.Reshape([1, 1, -1])

    def call(self, x):
        """

        Args:
        - x: input sequences

        Shape:
        - Inputs:
        - x: (B,L) where B is the batch size, L is the sequence length
        - Outputs:
        - x: (B,1,1,L) where B is the batch size, L is the input sequence length

        """

        x = self.lambda_layer(self.reshape_layer(x))

        return x


def add_reg_token(x, voc_size):
    """
    Tp and Ts Tokens Function: adds the Tp or the Ts token to the input sequences

    Args:
    - x: inputs sequences
    - voc_size [int]: number of unique tokens

    Shape:
    - Inputs:
    - x: (B,L) where B is the batch size, L is the sequence length
    - Outputs:
    - x: (B,1+L) where B is the batch size, L is the input sequence length

    """

    reg_token = tf.convert_to_tensor(voc_size + 1, dtype=tf.int32)
    broadcast_shape = tf.where([True, False], tf.shape(x), [0, 1])
    reg_token = tf.broadcast_to(reg_token, broadcast_shape)

    return tf.concat([reg_token, x], axis=1)




class reshape_bind_vector(tf.keras.layers.Layer):
    """
    Binding region-guided attention masking matrix

    Args:
    - inputs: [padding masking matrix, predicted 1D binding pocket]

    Shape:
    - Inputs:
    - Padding Masking Matrix: (B,1,1,L-1): where B is the batch size, L is the input sequence length
    - Predicted 1D Binding Pocket: (B,L,1): where B is the batch size, L is the input sequence length

    - Outputs:
    - bind_vector: (B,1,1,L-1): where B is the batch size, L is the input sequence length

    """
    def __init__(self, **kwargs):
        super(reshape_bind_vector, self).__init__(**kwargs)

        self.reshape_1 = tf.keras.layers.Reshape([-1], name='reshape_1')
        self.reshape_2 = tf.keras.layers.Reshape([1, 1, -1], name='reshape_2')

    def call(self, inputs):
        """

    Args:
    - inputs: [padding masking matrix, predicted 1D binding pocket]

    Shape:
    - Inputs:
    - Padding Masking Matrix: (B,1,1,L-1): where B is the batch size, L is the input sequence length
    - Predicted 1D Binding Pocket: (B,L,1): where B is the batch size, L is the input sequence length

    - Outputs:
    - bind_vector: (B,1,1,L-1): where B is the batch size, L is the input sequence length

        """
        pad_mask, bind_vector = inputs

        bind_vector = tf.cast(tf.math.sigmoid(bind_vector) > 0.5, tf.float32)

        zeros_mask = tf.math.equal(pad_mask, 0)
        zeros_mask = tf.squeeze(zeros_mask, axis=(1, 2))
        zeros_mask = tf.expand_dims(zeros_mask, axis=-1)

        bind_vector_zeros_mask = tf.cast(tf.math.logical_and(zeros_mask, tf.math.equal(bind_vector, 1)), tf.float32)

        # In the case where the binding vector has not any binding region detected
        bind_vector_zeros_mask = tf.reduce_sum(tf.squeeze(bind_vector_zeros_mask, axis=-1), axis=-1)
        bind_vector_zeros_mask = bind_vector_zeros_mask > 0
        bind_vector_zeros_mask = tf.expand_dims(bind_vector_zeros_mask, axis=1)
        bind_vector_zeros_mask = tf.repeat(bind_vector_zeros_mask, bind_vector.shape[1], axis=-1)
        bind_vector_zeros_mask = tf.expand_dims(bind_vector_zeros_mask, axis=-1)

        bind_vector_zeros_mask = tf.where(bind_vector_zeros_mask, bind_vector, tf.ones_like(bind_vector))

        bind_vector = self.reshape_2(self.reshape_1(bind_vector_zeros_mask))


        bind_vector = tf.where(tf.equal(bind_vector, 0), tf.ones_like(bind_vector),
                               tf.zeros_like(bind_vector))

        ones_mask = tf.math.equal(pad_mask, 1)
        bind_vector = tf.where(ones_mask, tf.ones_like(bind_vector), bind_vector)

        return bind_vector

# class reshape_bind_vector(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(reshape_bind_vector, self).__init__(**kwargs)
# 
#         self.reshape_1 = tf.keras.layers.Reshape([-1], name='reshape_1')
#         self.reshape_2 = tf.keras.layers.Reshape([1, 1, -1], name='reshape_2')
# 
#     def call(self, inputs):
#         pad_mask, bind_vector = inputs
# 
#         bind_vector = tf.cast(tf.math.sigmoid(bind_vector) > 0.5, tf.float32)
# 
#         bind_vector = self.reshape_2(self.reshape_1(bind_vector))
# 
#         bind_vector = tf.where(tf.equal(bind_vector, 0), tf.ones_like(bind_vector),
#                                tf.zeros_like(bind_vector))
# 
#         ones_mask = tf.math.equal(pad_mask, 1)
#         bind_vector = tf.where(ones_mask, tf.ones_like(bind_vector), bind_vector)
# 
#         return bind_vector


# Optimizer Configuration Function 
def opt_config(opt):
    opt_fn = None
    if opt[0] == 'radam':
        opt_fn = tfa.optimizers.RectifiedAdam(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                              beta_2=float(opt[3]), epsilon=float(opt[4]),
                                              weight_decay=float(opt[5]), total_steps=int(opt[6]),
                                              warmup_proportion=float(opt[7]),
                                              min_lr=float(opt[8]))
    elif opt[0] == 'adam':
        opt_fn = tf.keras.optimizers.Adam(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                          beta_2=float(opt[3]), epsilon=float(opt[4]))

    elif opt[0] == 'adamw':
        opt_fn = tfa.optimizers.AdamW(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                      beta_2=float(opt[3]), epsilon=float(opt[4]),
                                      weight_decay=float(opt[5]))

    elif opt[0] == 'lamb':
        opt_fn = tfa.optimizers.LAMB(learning_rate=float(opt[1]), beta_1=float(opt[2]),
                                     beta_2=float(opt[3]), epsilon=float(opt[4]))

    return opt_fn

    
# Function to gather the aggregated representation token and the remaining tokens    
def rearrange_tokens(tokens):
    cls_token = tf.expand_dims(tf.gather(tokens, 0, axis=1), axis=1)
    seq_tokens = tf.gather(tokens, tf.range(1, tokens.shape[1]), axis=1)

    return cls_token, seq_tokens


# Activation Configuration Function
def af_config(activation_fn):
    if activation_fn == 'gelu':
        activation_fn = gelu

    elif activation_fn == 'tanh':
        activation_fn = tf.math.tanh

    else:
        activation_fn = activation_fn

    return activation_fn
