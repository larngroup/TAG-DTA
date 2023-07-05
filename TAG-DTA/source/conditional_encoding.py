# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class cond_emb(tf.keras.layers.Layer):
    """
    Conditional Embedding Layer: Adds info to distinguish the aggregated representation of the SMILES strings from the protein tokens &
    updates the positional information of the concatenated DTI representations via learnable dictionary lookup matrices

    Args:
    - num_conditions [int]: number of conditions
    - max_len [int]: maximum length of the concatenated DTI representation
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout

    """

    def __init__(self, num_conditions, max_len, d_model, dropout_rate, **kwargs):
        super(cond_emb, self).__init__(**kwargs)

        self.num_conditions = num_conditions
        self.max_len = max_len
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.pos_enc_layer = tf.keras.layers.Embedding(self.max_len, self.d_model)
        self.condition_enc_layer = tf.keras.layers.Embedding(self.num_conditions + 1, self.d_model)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, cond_pos):
        """

        Args:
        - inputs: concatenated DTI representation
        - cond_pos: condition-based vector

        Shape:
        - Inputs:
        - inputs: (B,L,E): where B is the batch size, L is the input sequence length,
                        E is the embedding dimension
        - cond_pos: (1,L): L is the input sequence length

        - Outputs:
        - output_tensor: (B,L,E):  where B is the batch size, L is the input sequence length,
                        E is the embedding dimension

        """

        tgt_tensor = tf.range(self.max_len)
        cond_tensor = tf.zeros(shape=self.max_len)
        for i in range(len(cond_pos)):
            cond_tensor = tf.tensor_scatter_nd_update(cond_tensor, [[i]], [cond_pos[i]])

        output_tensor = inputs + self.pos_enc_layer(tgt_tensor) + self.condition_enc_layer(cond_tensor)
        output_tensor = self.dropout_layer(output_tensor)
        return output_tensor

    def get_config(self):
        config = super(cond_emb, self).get_config()
        config.update({
            'num_conditions': self.num_conditions,
            'max_len': self.max_len,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate})

        return config
