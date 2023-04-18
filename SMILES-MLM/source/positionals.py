# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class PositionalsLayer(tf.keras.layers.Layer):
    """
    Positional Embedding Layer: Adds info about the position of each token via a learnable dictionary lookup matrix

    Args:
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout
    - max_len [int]: number of input tokens (length)

    """

    def __init__(self, max_len, d_model, dropout_rate, **kwargs):
        super(PositionalsLayer, self).__init__(**kwargs)

        self.max_len = max_len
        self.d_model = d_model
        self.dropout_rate = dropout_rate


    def build(self, input_shape):
        self.pos_enc_layer = tf.keras.layers.Embedding(self.max_len, self.d_model)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs):
        """
        Shape:
        - Inputs:
        - inputs: (B,L,E): where B is the batch size, L is the sequence length,
                        E is the embedding dimension

        - Outputs:
        - output_tensor: (B,L,E): where B is the batch size, L is the sequence length,
                        E is the embedding dimension
        
        """

        tgt_tensor = tf.range(self.max_len)

        output_tensor = inputs + self.pos_enc_layer(tgt_tensor) 
        output_tensor = self.dropout_layer(output_tensor)
        return output_tensor

    def get_config(self):
        config = super(PositionalsLayer, self).get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate})

        return config
