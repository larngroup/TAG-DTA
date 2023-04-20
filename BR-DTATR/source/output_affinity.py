import tensorflow as tf


class affinity_fcnn(tf.keras.Model):
    
    """
    Fully-Connected Feed-Forward Network (FCNN)
    Args:
    - mlp_depth [int]: number of dense layers for the FCNN
    - mlp_units [list of ints]: number of hidden neurons for each one of the dense layers
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout
    
    """
    def __init__(self, mlp_depth, mlp_units, atv_fun, dropout_rate, **kwargs):
        super(affinity_fcnn, self).__init__(**kwargs)

        self.mlp_depth = mlp_depth
        self.mlp_units = mlp_units
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.output_dense = tf.keras.layers.Dense(units=1, name='mlp_out')
        self.layer_norm_concat= tf.keras.layers.LayerNormalization(epsilon=1e-5, name='norm_concat')
        if self.mlp_depth != 0:
            self.mlp_head = tf.keras.Sequential(name='mlp_dense')
            for i in range(self.mlp_depth):
                if self.mlp_depth == 1:
                    self.mlp_head.add(tf.keras.layers.Dense(self.mlp_units[i], activation=self.atv_fun))

                elif i != self.mlp_depth - 1:
                    self.mlp_head.add(tf.keras.layers.Dense(self.mlp_units[i], activation=self.atv_fun))
                    self.mlp_head.add(tf.keras.layers.Dropout(self.dropout_rate))
                else:
                    self.mlp_head.add(tf.keras.layers.Dense(self.mlp_units[i], activation=self.atv_fun))

    def call(self, inputs):

        """
        Args:
        - inputs: [Protein Representation, SMILES Represntation, Binding Region-Guided DTI Representation]

        Shape:
        - Inputs:
        - Protein Representation : (B,E_P) where B is the batch size, E_P is the embedding dimension for the Protein Representation

        - SMILES Representation : (B,E_S) where B is the batch size, E_S is the embedding dimension for the Protein Representation

        - Binding Region-Guided DTI Representation : (B,E_DTI) where B is the batch size,
                                                     E_DTI is the embedding dimension for the Binding Region-Guided DTI Representation


        - Outputs:
        - final_out: (B,1): where B is the batch size

        """

        out_input = self.layer_norm_concat(tf.concat(inputs, axis=-1))

        if self.mlp_depth != 0:
            fc_outs = self.mlp_head(out_input)
        else:
            fc_outs = out_input

        final_out = self.output_dense(fc_outs)

        return final_out

    def get_config(self):
        config = super(affinity_fcnn, self).get_config()
        config.update({
            'mlp_depth': self.mlp_depth,
            'mlp_units': self.mlp_units,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate,
            'output_dense': self.output_dense,
            'mlp_head': self.mlp_head})
        return config
