# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import tensorflow as tf


class bert_pooler(tf.keras.layers.Layer):

    """
    Position-Wise Pooling Block to map the aggregated representation of the SMILES Strings
    to the last dimension of the protein tokens (embedding/representation size)

    Args:
    - dense_size [int]: number of hidden units of the projection dense layer
    - atv_fun: dense layer activation function
    - dense_opt [int]: 1 - project the last dimension of the aggregated representation

    """

    def __init__(self, dense_size, atv_fun, dense_opt, **kwargs):

        super(bert_pooler, self).__init__(**kwargs)

        self.dense_size = dense_size
        self.atv_fun = atv_fun
        self.dense_opt = dense_opt

    def build(self, input_shape):
        if bool(self.dense_opt):
            self.dense_layer = tf.keras.layers.Dense(units=self.dense_size, activation=self.atv_fun)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.reshape_layer = tf.keras.layers.Reshape((1, -1))
        self.lambda_layer = tf.keras.layers.Lambda((lambda x: tf.gather(x, 0, axis=1)))

    def call(self, hidden_inputs):
        """

        Args:
        - hidden_inputs: SMILES Transformer-Encoder Outputs

        Shape:
        - Inputs:
        - hidden_inputs: (B,L,E): where B is the batch size, L is the SMILES sequence length,
                        E is the embedding dimension

        - Outputs:
        - cls_token: (B,1,E_Proj):  where B is the batch size,
                                    E_Proj is the projected dimension (protein token representation dimension)

        """

        cls_token = self.lambda_layer(hidden_inputs)
        cls_token = self.reshape_layer(cls_token)

        if bool(self.dense_opt):
            cls_token = self.layernorm(self.dense_layer(cls_token))
        return cls_token

    def get_config(self):
        config = super(bert_pooler, self).get_config()
        config.update({
            'dense_size': self.dense_size,
            'atv_fun': self.atv_fun,
            'dense_opt': self.dense_opt})

        return config
