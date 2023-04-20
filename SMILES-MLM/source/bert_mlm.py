import tensorflow as tf
from transformer_encoder import *
from embedding_layer import *
from argument_parser import *
import time
from dataset_builder_util import *
import os
from bert_mlm_train_util import *
import itertools
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import math
from positionals import *

def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


def bert_mlm(FLAGS, d_model, dropout_rate, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dim_k, parameter_sharing):

    """
    Function to build the SMILES Pre-Train MLM Model
    Args:
    - FLAGS: arguments object
    - d_model [int]: embedding dim
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


    if enc_atv_fun == 'gelu':
        enc_atv_fun = gelu

    inputs = tf.keras.Input(shape=FLAGS.smiles_len+1, dtype=tf.int64, name='input_layer')
    pad_mask = attn_pad_mask()(inputs)
    input_embedding = tf.keras.layers.Embedding(FLAGS.smiles_dict_len + 3, d_model, name='emb_layer')(inputs)
    char_embedding = PositionalsLayer(FLAGS.smiles_len + 1, d_model, dropout_rate, name='pos_layer')(input_embedding)
    # char_embedding = EmbeddingLayer(FLAGS.smiles_dict_len+3, d_model, dropout_rate, FLAGS.pos_enc_option,
    #                                  name='emb_layer')(inputs)  # Sine and Cosine Positional Embedding

    trans_encoder,_ = Encoder(d_model, num_enc_layers, num_enc_heads, enc_dff, enc_atv_fun, dropout_rate, dim_k, parameter_sharing,
                             FLAGS.full_attn, FLAGS.return_intermediate, name='trans_encoder')(char_embedding, pad_mask)


    outputs = tf.keras.layers.Dense(units=FLAGS.smiles_dict_len+3, name='mlm_cls')(trans_encoder)

    return tf.keras.Model(inputs=inputs,outputs=outputs)


def pre_train(FLAGS, data_train, data_val, model_function):

    """
    MLM Pre-Train Grid Search function
    Args:
    - FLAGS: arguments object
    - data_train: [smiles_mlm_train_data, smiles_mlm_train_target, smiles_mlm_train_weights]
    - data_val: [smiles_mlm_val_data, smiles_mlm_val_target, smiles_mlm_val_weights]
    - model_function: function that creates the model
    """


    d_model_set = FLAGS.d_model
    dropout_rate_set = FLAGS.dropout_rate
    enc_layers = FLAGS.transformer_depth
    heads_set = FLAGS.transformer_heads
    dff_set = FLAGS.d_ff_dim
    atv_fn_set = FLAGS.dense_atv_fun
    epochs_set = FLAGS.num_epochs
    batch_set = FLAGS.batch_dim
    opt_set = FLAGS.optimizer_fn
    param_sharing_set = FLAGS.parameter_sharing
    dim_k_set = FLAGS.dim_k



    for params in itertools.product(d_model_set,dropout_rate_set,enc_layers,heads_set,
                                    dff_set,atv_fn_set,epochs_set,batch_set, opt_set, dim_k_set, param_sharing_set):

        FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
        FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"

        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)

        logging(str(FLAGS), FLAGS)

        logging("--------------------Grid Search-------------------", FLAGS)

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = params


        if p9[0] == 'radam':
            p9 = tfa.optimizers.RectifiedAdam(learning_rate=float(p9[1]), beta_1=float(p9[2]),
                                               beta_2=float(p9[3]), epsilon=float(p9[4]),
                                               weight_decay=float(p9[5]), total_steps=512500,
                                      warmup_proportion=0.01,
                                      min_lr=1e-05)
        elif p9[0] == 'adam':
            p9 = tf.keras.optimizers.Adam(learning_rate=float(p9[1]), beta_1=float(p9[2]),
                                           beta_2=float(p9[3]), epsilon=float(p9[4]))

        elif p9[0] == 'adamw':
            p9 = tfa.optimizers.AdamW(learning_rate=float(p9[1]), beta_1=float(p9[2]),
                                       beta_2=float(p9[3]), epsilon=float(p9[4]),
                                       weight_decay=float(p9[5]))

        logging(
            "D Model = %d, DropR = %0.4f, Enc Depth = %d, Heads = %d, DFF = %d, ATV Fn = %s, Epochs = %d,  Batch = %d, Optimizer = %s, Dim K = %d, Param Share = %s" %
            (p1, p2, p3, p4, p5, p6, p7, p8, p9.get_config(), p10, p11), FLAGS)


        data_train_v2 = data_train.batch(p8)
        data_val_v2 = data_val.batch(p8)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')


        bert_model = model_function(FLAGS, p1, p2, p3, p4, p5, p6, p10, p11)


        ckpt_obj = tf.train.Checkpoint(step=tf.Variable(1), optimizer=p9, model=bert_model)

        ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt_obj, directory=FLAGS.checkpoint_path,
                                                        max_to_keep=1)

        run_train_val(FLAGS, p7, data_train_v2,data_val_v2,bert_model,p9,loss_fn,train_loss,train_accuracy, val_loss, val_accuracy,
                      ckpt_obj, ckpt_manager)



def run_pre_train(FLAGS):
    """
    Run MLM Pre-Train Grid Search function
    Args:
    - FLAGS: arguments object
    """

    dataset_train, dataset_val = train_val(load_data_mask(), 0.8)

    model_fn = bert_mlm

    pre_train(FLAGS, dataset_train, dataset_val, model_fn)


if __name__ == '__main__':
    FLAGS = argparser()
    run_pre_train(FLAGS)





















