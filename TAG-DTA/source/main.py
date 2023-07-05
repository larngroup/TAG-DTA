import os
from args_parser import *
from transformer_encoder import *
import time
import itertools
from train_util import *
from dataset_builder_util import *
from output_affinity import *
from output_binding import *
from bert_pooler import *
from positional_encoding import *
from bert_smiles import *
from bert_encoder import *
from conditional_encoding import *
from itertools import combinations_with_replacement
from utils import *
from train_util import *

def build_bert_binding_affinity(FLAGS, prot_emb_size, bert_smiles_train,
                                prot_enc_depth, prot_enc_heads, prot_enc_dff, prot_atv_fun,
                                drop_rate, smiles_pooler_dense_opt, smiles_pooler_atv, bind_enc_depth,
                                bind_enc_heads, bind_enc_dff, bind_enc_atv_fun, bind_fc_depth,
                                bind_fc_units, bind_fc_atv_fun,
                                affinity_enc_depth, affinity_enc_heads, affinity_enc_dff,
                                affinity_enc_atv_fun, affinity_fc_depth, affinity_fc_units,
                                affinity_fc_atv_fun):
    """
    Function to build the TAG-DTA Model

    Args:
    - FLAGS: arguments object
    - prot_emb_size [int] : protein embedding size
    - bert_smiles_train[int]: 1 - train SMILES Pre-trained Transformer-Encoder, 0 - Freeze Pre-trained Transformer-Encoder Weights
    - prot_enc_depth [int]: number of Protein Transformer-Encoders
    - prot_enc_heads [int]: number of heads for the Protein MHSA
    - prot_enc_dff [int]: hidden numbers for the first dense layer of the Protein PWFFN 
    - prot_atv_fun: Protein Trasnformer-Encoder dense layers activation function
    - drop_rate [float]: % of dropout
    - smiles_pooler_dense_opt [int]: 1-project last dimension of the aggregated representation of the SMILES strings
    - smiles_pooler_atv: PWPool activation model_function
    - bind_enc_depth [int]: number of Binding Pocket Transformer-Encoders
    - bind_enc_heads [int]:  number of heads for the Binding Pocket MHSA
    - bind_enc_dff [int]: hidden numbers for the first dense layer of the Binding Pocket PWFFN 
    - bind_enc_atv_fun: Binding Pocket Transformer-Encoder dense layers activation function
    - bind_fc_depth [int]: Binding Pocket PWMLP number of layers
    - bind_fc_units [list of ints]: hidden neurons for each one of the dense layers of the Binding Pocket PWMLP
    - bind_fc_atv_fun: Binding Pocket PWMLP activation function

    - affinity_enc_depth [int]: number of Binding Affinity Transformer-Encoders
    - affinity_enc_heads [int]:  number of heads for the Binding Affinity MHSA
    - affinity_enc_dff [int]: hidden numbers for the first dense layer of the Binding Affinity PWFFN 
    - affinity_enc_atv_fun: Binding Affinity Transformer-Encoder dense layers activation function
    - affinity_fc_depth [int]: Binding Affinity FCNN number of layers
    - affinity_fc_units [list of ints]: hidden neurons for each one of the dense layers of the Binding Affinity FCNN
    - affinity_fc_atv_fun: Binding Affinity FCNN activation function

    Outputs:
    - TAG-DTA Model
    """

    prot_atv_fun = af_config(prot_atv_fun)
    smiles_pooler_atv = af_config(smiles_pooler_atv)
    bind_enc_atv_fun = af_config(bind_enc_atv_fun)
    bind_fc_atv_fun = af_config(bind_fc_atv_fun)
    affinity_enc_atv_fun = af_config(affinity_enc_atv_fun)
    affinity_fc_atv_fun = af_config(affinity_fc_atv_fun)

    # Model Input
    prot_input = tf.keras.Input(FLAGS.prot_len + 1, dtype=tf.int64, name='prot_input')
    smiles_input = tf.keras.Input(FLAGS.smiles_len + 1, dtype=tf.int64, name='smi_input')

    # Protein & SMILES Mask
    prot_mask = attn_pad_mask(name='prot_mask_layer')(prot_input)
    smiles_mask = attn_pad_mask(name='smiles_mask_layer')(smiles_input)

    # SMILES Pre-trained Transformer Encoder
    smiles_model = load_bert_smiles(bert_smiles_mlm(*FLAGS.bert_smiles_model_dict), FLAGS.bert_smiles_model_ckpt_obj)

    if bool(bert_smiles_train):
        smiles_model_out, _ = smiles_model(smiles_input)
    else:
        smiles_model.trainable = False
        smiles_model_out, _ = smiles_model(smiles_input)

    # Protein Transformer-Encoder
    prot_bert_encoder = bert_encoder(FLAGS.prot_len + 1, FLAGS.prot_dict_len + 3, FLAGS.prot_enc_full_attention,
                                     FLAGS.prot_enc_return_interm, prot_emb_size, drop_rate, prot_enc_depth,
                                     prot_enc_heads, prot_enc_dff, prot_atv_fun, FLAGS.prot_enc_dim_k,
                                     FLAGS.prot_enc_param_share)

    prot_bert_encoder._name = 'prot_bert_encoder'

    prot_bert_tokens, _ = prot_bert_encoder(prot_input)

    # Binding Vector Prediction

    prot_cls, prot_tokens = rearrange_tokens(prot_bert_tokens)

    smiles_cls = bert_pooler(prot_emb_size, smiles_pooler_atv, smiles_pooler_dense_opt,
                            name='bert_smiles_pooler')(smiles_model_out)

    smi_cls_prot_tokens = tf.keras.layers.Concatenate(axis=1, name='smiles_prot_concat')([smiles_cls, prot_tokens])

    cond_positions = [1] + [2] * FLAGS.prot_len

    combined_info = cond_emb(2, FLAGS.prot_len + 1, prot_emb_size, drop_rate,
                                 name='pos_layer')(smi_cls_prot_tokens, cond_positions)

    bind_enc_tokens, _ = transformer_encoder(prot_emb_size, bind_enc_depth, bind_enc_heads, bind_enc_dff, bind_enc_atv_fun,
                                 drop_rate, FLAGS.bind_enc_dim_k, FLAGS.bind_enc_param_share,
                                 FLAGS.bind_enc_full_attention,
                                 FLAGS.bind_enc_return_interm,
                                 name='binding_vector_encoder')(combined_info, prot_mask)

    bind_vector = binding_pocket_pwmlp(bind_fc_depth, bind_fc_units, bind_fc_atv_fun,
                             drop_rate, name='binding_output_block')(bind_enc_tokens)

    # Binding Affinity Prediction

    affinity_input = combined_info

    affinity_mask = reshape_bind_vector()(
        [tf.gather(prot_mask, tf.range(1, prot_mask.shape[-1]), axis=-1), bind_vector])

    affinity_mask = tf.concat([tf.expand_dims(tf.gather(prot_mask, 0, axis=-1) * 0, axis=-1), affinity_mask], axis=-1)

    affinity_enc_tokens, _ =  transformer_encoder(prot_emb_size, affinity_enc_depth, affinity_enc_heads,
                                     affinity_enc_dff, affinity_enc_atv_fun,
                                     drop_rate, FLAGS.affinity_enc_dim_k, FLAGS.affinity_enc_param_share,
                                     FLAGS.affinity_enc_full_attention,
                                     FLAGS.affinity_enc_return_interm,
                                     name='affinity_vector_encoder')(affinity_input, affinity_mask)

    affinity_fc_input = [tf.squeeze(prot_cls, axis=1), tf.squeeze(smiles_cls, axis=1),
                         tf.gather(affinity_enc_tokens, 0, axis=1)]

    affinity_value = affinity_fcnn(affinity_fc_depth, affinity_fc_units, affinity_fc_atv_fun,
                                 drop_rate, name='affinity_output_block')(affinity_fc_input)

    model = tf.keras.Model(inputs=[prot_input, smiles_input], outputs=[bind_vector, affinity_value])

    return model


def grid_search(FLAGS, data_bind_vector, data_bind_aff, model_function):
    """
    TAG-DTA Grid Search function

    Args:
    - FLAGS: arguments object
    - data_bind_vector: [bind_data_train, bind_data_val]
    - data_bind_aff: [affinity_protein, affinity_smiles, affinity_kd, affinity_folds]
    - model_function: function that creates the model
    """

    prot_emb_dff_list = []
    for i in FLAGS.prot_emb_dff:
        for j in range(len(i) - 1):
            prot_emb_dff_list.append([i[0], i[j + 1]])
    # prot_emb_size_list = FLAGS.prot_emb_size
    bert_smiles_train_list = FLAGS.bert_smiles_train
    prot_enc_depth_list = FLAGS.prot_enc_depth
    prot_enc_heads_list = FLAGS.prot_enc_heads
    # prot_enc_dff_list = FLAGS.prot_enc_dff
    prot_enc_af_list = FLAGS.prot_atv_fun
    drop_rate_list = FLAGS.dropout_rate
    smiles_pooler_dense_opt_list = FLAGS.smiles_pooler_dense_opt
    smiles_pooler_af_list = FLAGS.smiles_pooler_atv
    bind_enc_depth_list = FLAGS.bind_enc_depth
    bind_enc_heads_list = FLAGS.bind_enc_heads
    # bind_enc_dff_list = [FLAGS.prot_emb_dff[i + 1] for i in range(len(FLAGS.prot_emb_dff) - 1)]
    # bind_enc_dff_list = FLAGS.bind_enc_dff
    bind_enc_af_list = FLAGS.bind_enc_atv_fun
    # bind_fc_depth_units_list = [[FLAGS.bind_fc_depth[0], FLAGS.bind_fc_units]]
    bind_fc_depth_units_list = []
    for i in FLAGS.bind_fc_depth:
        for k in combinations_with_replacement(FLAGS.bind_fc_units, i):
            bind_fc_depth_units_list.append([i, list(k)])

    # bind_fc_depth_list = FLAGS.bind_fc_depth
    # bind_fc_units_list = FLAGS.bind_fc_units

    bind_fc_af_list = FLAGS.bind_fc_atv_fun
    affinity_enc_depth_list = FLAGS.affinity_enc_depth
    affinity_enc_heads_list = FLAGS.affinity_enc_heads
    # affinity_enc_dff_list = FLAGS.affinity_enc_dff
    # affinity_enc_dff_list = [FLAGS.prot_emb_dff[i + 1] for i in range(len(FLAGS.prot_emb_dff) - 1)]
    affinity_enc_af_list = FLAGS.affinity_enc_atv_fun

    # affinity_fc_depth_units_list = [[FLAGS.affinity_fc_depth[0], FLAGS.affinity_fc_units]]
    affinity_fc_depth_units_list = []
    for i in FLAGS.affinity_fc_depth:
        for k in combinations_with_replacement(FLAGS.affinity_fc_units, i):
            affinity_fc_depth_units_list.append([i, list(k)])

    # affinity_fc_depth_list = FLAGS.affinity_fc_depth
    # affinity_fc_units_list = FLAGS.affinity_fc_units

    affinity_fc_af_list = FLAGS.affinity_fc_atv_fun
    batch_list = FLAGS.batch_size
    epochs_list = FLAGS.epoch_num
    pre_train_epochs_list = FLAGS.pre_train_epochs
    bind_vector_epochs_list = FLAGS.bind_vector_epochs
    bind_affinity_epochs_list = FLAGS.bind_affinity_epochs
    smiles_bert_opt_list = FLAGS.smiles_bert_opt
    binding_opt_list = FLAGS.binding_bert_opt
    affinity_opt_list = FLAGS.affinity_bert_opt
    binding_loss_opt = FLAGS.bind_loss_opt

    # for params in itertools.product(prot_emb_size_list, bert_smiles_train_list, prot_enc_depth_list,
    #                                 prot_enc_heads_list, prot_enc_dff_list, prot_enc_af_list, drop_rate_list,
    #                                 smiles_pooler_dense_opt_list, smiles_pooler_af_list, bind_enc_depth_list,
    #                                 bind_enc_heads_list, bind_enc_dff_list, bind_enc_af_list, bind_fc_depth_list,
    #                                 bind_fc_units_list, bind_fc_af_list, affinity_enc_depth_list,
    #                                 affinity_enc_heads_list,
    #                                 affinity_enc_dff_list, affinity_enc_af_list, affinity_fc_depth_list,
    #                                 affinity_fc_units_list,
    #                                 affinity_fc_af_list, batch_list, epochs_list, pre_train_epochs_list,
    #                                 bind_vector_epochs_list,
    #                                 bind_affinity_epochs_list, smiles_bert_opt_list, binding_opt_list,
    #                                 affinity_opt_list, binding_loss_opt):

    for params in itertools.product(prot_emb_dff_list, bert_smiles_train_list, prot_enc_depth_list,
                                    prot_enc_heads_list, prot_enc_af_list, drop_rate_list,
                                    smiles_pooler_dense_opt_list, smiles_pooler_af_list, bind_enc_depth_list,
                                    bind_enc_heads_list, bind_enc_af_list, bind_fc_depth_units_list,
                                    bind_fc_af_list, affinity_enc_depth_list,
                                    affinity_enc_heads_list,
                                    affinity_enc_af_list, affinity_fc_depth_units_list,
                                    affinity_fc_af_list, batch_list, epochs_list, pre_train_epochs_list,
                                    bind_vector_epochs_list,
                                    bind_affinity_epochs_list, smiles_bert_opt_list, binding_opt_list,
                                    affinity_opt_list, binding_loss_opt):

        # p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32 = params

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27 = params

        FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
        FLAGS.checkpoint_dir = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"

        if not os.path.exists(FLAGS.log_dir):
            os.makedirs(FLAGS.log_dir)
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)

        fold_metrics = []
        folds = data_bind_aff[-1]
        for fold_idx in range(len(folds)):

            logger(FLAGS, fold_idx, FLAGS)
            logger("--------------------Grid Search-------------------", fold_idx, FLAGS)
            logger("-----------------Affinity Fold: " + str(fold_idx) + "-----------------", fold_idx, FLAGS)

            checkpoint_dir_fold = FLAGS.checkpoint_dir + str(fold_idx) + '/'

            if not os.path.exists(checkpoint_dir_fold):
                os.makedirs(checkpoint_dir_fold)

            # SMILES Transformer-Encoder Optimizer
            p24_v2 = opt_config(p24)

            # Binding Vector Prediction Optimizer
            p25_v2 = opt_config(p25)

            # Binding Affinity Prediction Optimizer
            p26_v2 = opt_config(p26)

            logger(("Prot Emb Size: %d, BERT SMILES Train: %d, Prot Enc Depth: %d, Prot Enc Heads: %d, Prot Enc DFF: "
                    "%d, " +
                    "Prot Enc AF: %s, Drop Rate: %0.4f, SMILES Pooler Dense Opt: %d, SMILES Pooler AF: %s, Bind Enc Depth: %d, "
                    +
                    "Bind Enc Heads: %d, Bind Enc DFF: %d, Bind Enc AF: %s, Bind FC Depth: %d, Bind FC Units: %s, "
                    "Bind FC AF: %s, " +
                    "Affinity Enc Depth: %d, Affinity Enc Heads: %d, Affinity Enc DFF: %d, Affinity Enc AF: %s, "
                    "Affinity FC Depth: %d, " +
                    "Affinity FC Units: %s, Affinity FC AF: %s, Batch: %d, Epochs: %d, Bind Pre-Train Epochs: %d, "
                    "Bind Epochs Ratio: %d, " +
                    "Affinity Epochs Ratio: %d, SMILES Bert Optimizer: %s, Bind Vector Optimizer: %s, Affinity "
                    "Optimizer: %s, Bind Loss Opt: %s, " +
                    "Fold: %d") %
                   (p1[0], p2, p3, p4, p1[-1], p5, p6, p7, p8, p9, p10, p1[-1], p11, p12[0], p12[1], p13, p14, p15,
                    p1[-1], p16, p17[0], p17[1], p18, p19, p20, p21, p22, p23, p24_v2.get_config(), p25_v2.get_config(),
                    p26_v2.get_config(), p27, fold_idx), fold_idx, FLAGS)

            # Binding Vector Prediction - Hold-out Validation
            with tf.device('/cpu:0'):
                bind_data_train, bind_data_val = data_bind_vector
                bind_data_train = bind_data_train.batch(p19).take(2)
                bind_data_val = bind_data_val.batch(p19).take(2)

            # Binding Pocket Loss Function
            if p27[0] == 'focal':
                bind_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True,
                                                                       gamma=float(p27[1]),
                                                                       reduction=tf.keras.losses.Reduction.NONE)

            elif p27[0] == 'standard':
                bind_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)

            # Binding Pocket Training and Validation Objects
            bind_train_loss = tf.keras.metrics.Mean(name='bind_train_loss')
            bind_train_accuracy = tf.keras.metrics.Mean(name='bind_train_acc')
            bind_train_f1 = tf.keras.metrics.Mean(name='bind_train_f1')
            bind_train_recall = tf.keras.metrics.Mean(name='bind_train_recall')
            bind_train_precision = tf.keras.metrics.Mean(name='bind_train_precision')
            bind_train_mcc = tf.keras.metrics.Mean(name='bind_train_mcc')

            bind_val_loss = tf.keras.metrics.Mean(name='bind_val_loss')
            bind_val_accuracy = tf.keras.metrics.Mean(name='bind_val_acc')
            bind_val_f1 = tf.keras.metrics.Mean(name='bind_val_f1')
            bind_val_recall = tf.keras.metrics.Mean(name='bind_val_recall')
            bind_val_precision = tf.keras.metrics.Mean('bind_val_precision')
            bind_val_mcc = tf.keras.metrics.Mean(name='bind_val_mcc')

            # Binding Affinity Prediction - 5-Fold Cross Validation
            with tf.device('/cpu:0'):
                index_train = list(
                    itertools.chain.from_iterable([folds[i] for i in range(len(folds)) if i != fold_idx]))

                index_val = folds[fold_idx]
                affinity_data_train = [tf.gather(i, index_train) for i in data_bind_aff[:-1]]
                affinity_data_val = [tf.gather(i, index_val) for i in data_bind_aff[:-1]]

                affinity_data_train = tf.data.Dataset.from_tensor_slices(tuple(affinity_data_train))
                # data_train = data_train.shuffle(buffer_size=len(data_train), reshuffle_each_iteration=False)
                affinity_data_train = affinity_data_train.batch(p19).take(2)

                affinity_data_val = tf.data.Dataset.from_tensor_slices(tuple(affinity_data_val))
                affinity_data_val = affinity_data_val.batch(p19).take(2)

            # Binding Affinity Loss Function, Training and Validation Objects
            affinity_loss_fn = tf.keras.losses.MeanSquaredError()

            affinity_train_loss = tf.keras.metrics.Mean('train_loss')
            affinity_train_rmse = tf.keras.metrics.Mean('train_rmse')
            affinity_train_ci = tf.keras.metrics.Mean('train_ci')

            affinity_val_loss = tf.keras.metrics.Mean('val_loss')
            affinity_val_rmse = tf.keras.metrics.Mean('val_rmse')
            affinity_val_ci = tf.keras.metrics.Mean('val_ci')

            # Binding Vector + Binding Affinity Model
            # model = model_function(FLAGS, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
            #                        p15, p16, p17, p18, p19, p20, p21, p22, p23)

            model = model_function(FLAGS, p1[0], p2, p3, p4, p1[-1], p5, p6, p7, p8, p9, p10, p1[-1],
                                   p11, p12[0], p12[-1], p13, p14, p15, p1[-1], p16, p17[0], p17[-1], p18)

            # Binding Vector Final Dense Layer Initial Bias (Imbalanced Data Correction) np.log(pos/neg)
            # initial_bias = np.array([-2.4664805])
            initial_bias = np.array([-1.663877])

            model.get_layer('binding_output_block').get_layer('mlp_out').set_weights(
                [model.get_layer('binding_output_block').get_layer('mlp_out').get_weights()[0]] + [initial_bias])

            # Model Summary
            model.summary()

            # Checkpoint Object and Checkpoint Manager
            global_var = tf.Variable(1)
            ckpt_obj = tf.train.Checkpoint(step=global_var, smiles_bert_opt=p24_v2, bind_opt=p25_v2,
                                           affinity_opt=p26_v2, model=model)

            ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt_obj, directory=checkpoint_dir_fold,
                                                      max_to_keep=1)

            # Run Grid Search: Binding Vector - Hold-out Validation, Binding Affinity. Chemogenomic 5-Fold Cross-Val
            results = run_train_val(FLAGS, fold_idx, p20, bind_data_train, bind_data_val, affinity_data_train,
                                    affinity_data_val,
                                    model, p24_v2, p25_v2, p26_v2, p27, bind_loss_fn, bind_train_loss,
                                    bind_train_accuracy, bind_train_recall, bind_train_precision, bind_train_f1,
                                    bind_train_mcc, bind_val_loss, bind_val_accuracy, bind_val_recall,
                                    bind_val_precision,
                                    bind_val_f1, bind_val_mcc, affinity_loss_fn, affinity_train_loss,
                                    affinity_train_rmse,
                                    affinity_train_ci, affinity_val_loss, affinity_val_rmse, affinity_val_ci,
                                    ckpt_obj, ckpt_manager, p21, p22, p23)

            fold_metrics.append(results)

        fold_metrics = tuple(np.mean(np.array(fold_metrics), axis=0))

        logger("--------------Folds Metrics--------------", 'avg', FLAGS)
        logger(FLAGS, 'avg', FLAGS)
        logger(("Bind Vector Val Loss: %0.4f, Bind Vector Val ACC: %0.4f, Bind Vector Val Recall: %0.4f, " +
                "Bind Vector Val Precision: %0.4f, Bind Vector Val F1: %0.4f, Bind Vector Val MCC: %0.4f, " +
                "Affinity Val Loss: %0.4f, Affinity Val RMSE: %0.4f, Affinity Val CI: %0.4f") % fold_metrics, 'avg',
               FLAGS)


def run_grid_search(FLAGS):
    """
    Run TAG-DTA Grid Search function
    Args:
    - FLAGS: arguments object
    """

    with tf.device('/cpu:0'):
        # Binding Affinity Dataset
        affinity_data_path = {'data': '../data/affinity_data/davis_dataset_processed.csv',
                              'prot_dict': '',
                              'smiles_dict': '../dictionary/smiles_chembl_dict.txt',
                              'clusters': glob.glob('../data/affinity_data/clusters/*'),
                              'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                           '../dictionary/subword_units_map_uniprot.csv'],
                              'smiles_bpe': ''}

        affinity_protein, affinity_smiles, affinity_kd = dataset_builder(affinity_data_path).transform_dataset(1, 0,
                                                                                                               'Sequence',
                                                                                                               'SMILES',
                                                                                                               'Kd',
                                                                                                               FLAGS.prot_len,
                                                                                                               0,
                                                                                                               0,
                                                                                                               FLAGS.smiles_len)

        affinity_protein = add_reg_token(affinity_protein, FLAGS.prot_dict_len)
        affinity_smiles = add_reg_token(affinity_smiles, FLAGS.smiles_dict_len)

        _, _, _, clusters, _, _, _, _ = dataset_builder(affinity_data_path).get_data()

        affinity_folds = [list(clusters[i][1].iloc[:, 0]) for i in range(len(clusters)) if clusters[i][0] != 'test']
        affinity_data = [affinity_protein, affinity_smiles, affinity_kd, affinity_folds]

        # Binding Pocket Dataset
        bi_data = shuffle_split(load_data(FLAGS), 0.9)

    model_fn = build_bert_binding_affinity

    grid_search(FLAGS, bi_data, affinity_data, model_fn)


def run_train_model(FLAGS):
    print(FLAGS)
    with tf.device('/cpu:0'):
        # Binding Affinity Dataset
        affinity_data_path = {'data': '../data/affinity_data/davis_dataset_processed.csv',
                              'prot_dict': '',
                              'smiles_dict': '../dictionary/smiles_chembl_dict.txt',
                              'clusters': glob.glob('../data/affinity_data/clusters/*'),
                              'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                           '../dictionary/subword_units_map_uniprot.csv'],
                              'smiles_bpe': ''}

        affinity_protein, affinity_smiles, affinity_kd = dataset_builder(affinity_data_path).transform_dataset(1, 0,
                                                                                                               'Sequence',
                                                                                                               'SMILES',
                                                                                                               'Kd',
                                                                                                               FLAGS.prot_len,
                                                                                                               0,
                                                                                                               0,
                                                                                                               FLAGS.smiles_len)

        affinity_protein = add_reg_token(affinity_protein, FLAGS.prot_dict_len)
        affinity_smiles = add_reg_token(affinity_smiles, FLAGS.smiles_dict_len)

        _, _, _, clusters, _, _, _, _ = dataset_builder(affinity_data_path).get_data()

        affinity_folds = [list(clusters[i][1].iloc[:, 0]) for i in range(len(clusters)) if clusters[i][0] != 'test']
        affinity_folds = [list(clusters[i][1].iloc[:, 0]) for i in range(len(clusters)) if
                          clusters[i][0] == 'test'] + affinity_folds
        affinity_data = [affinity_protein, affinity_smiles, affinity_kd]

        index_train = list(
            itertools.chain.from_iterable([affinity_folds[i] for i in range(len(affinity_folds)) if i != 0]))

        index_val = affinity_folds[0]
        affinity_data_train = [tf.gather(i, index_train) for i in affinity_data]
        affinity_data_val = [tf.gather(i, index_val) for i in affinity_data]

        affinity_data_train = tf.data.Dataset.from_tensor_slices(tuple(affinity_data_train))
        # data_train = data_train.shuffle(buffer_size=len(data_train), reshuffle_each_iteration=False)
        affinity_data_train = affinity_data_train.batch(FLAGS.batch_size[0]).take(2)

        affinity_data_val = tf.data.Dataset.from_tensor_slices(tuple(affinity_data_val))
        affinity_data_val = affinity_data_val.batch(FLAGS.batch_size[0]).take(2)


        # Binding Pocket Dataset
        bind_data_train = load_data(FLAGS)
        bind_data_val = load_data(FLAGS, '../data/bind_data/coach_test/prot.tfrecords',
                                '../data/bind_data/coach_test/smiles.tfrecords',
                                '../data/bind_data/coach_test/target.tfrecords',
                                '../data/bind_data/coach_test/weights.tfrecords')

        bind_data_train = bind_data_train.batch(FLAGS.batch_size[0]).take(2)
        bind_data_val = bind_data_val.batch(FLAGS.batch_size[0]).take(2)

    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_dir = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # Binding Pocket Loss Function
    bind_loss_opt = FLAGS.bind_loss_opt[0]

    if bind_loss_opt[0] == 'focal':
        bind_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True,
                                                               gamma=float(bind_loss_opt[0][1]),
                                                               reduction=tf.keras.losses.Reduction.NONE)

    elif bind_loss_opt[0] == 'standard':
        bind_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)

    # Binding Pocket Training and Validation Objects
    bind_train_loss = tf.keras.metrics.Mean(name='bind_train_loss')
    bind_train_accuracy = tf.keras.metrics.Mean(name='bind_train_acc')
    bind_train_f1 = tf.keras.metrics.Mean(name='bind_train_f1')
    bind_train_recall = tf.keras.metrics.Mean(name='bind_train_recall')
    bind_train_precision = tf.keras.metrics.Mean(name='bind_train_precision')
    bind_train_mcc = tf.keras.metrics.Mean(name='bind_train_mcc')

    bind_val_loss = tf.keras.metrics.Mean(name='bind_val_loss')
    bind_val_accuracy = tf.keras.metrics.Mean(name='bind_val_acc')
    bind_val_f1 = tf.keras.metrics.Mean(name='bind_val_f1')
    bind_val_recall = tf.keras.metrics.Mean(name='bind_val_recall')
    bind_val_precision = tf.keras.metrics.Mean('bind_val_precision')
    bind_val_mcc = tf.keras.metrics.Mean(name='bind_val_mcc')

    # Binding Affinity Loss Function, Training and Validation Objects
    affinity_loss_fn = tf.keras.losses.MeanSquaredError()

    affinity_train_loss = tf.keras.metrics.Mean('train_loss')
    affinity_train_rmse = tf.keras.metrics.Mean('train_rmse')
    affinity_train_ci = tf.keras.metrics.Mean('train_ci')

    affinity_val_loss = tf.keras.metrics.Mean('val_loss')
    affinity_val_rmse = tf.keras.metrics.Mean('val_rmse')
    affinity_val_ci = tf.keras.metrics.Mean('val_ci')

    model = build_bert_binding_affinity(FLAGS, FLAGS.prot_emb_size[0], FLAGS.bert_smiles_train[0],
                                        FLAGS.prot_enc_depth[0], FLAGS.prot_enc_heads[0],
                                        FLAGS.prot_enc_dff[0], FLAGS.prot_atv_fun[0],
                                        FLAGS.dropout_rate[0], FLAGS.smiles_pooler_dense_opt[0],
                                        FLAGS.smiles_pooler_atv[0], FLAGS.bind_enc_depth[0],
                                        FLAGS.bind_enc_heads[0], FLAGS.bind_enc_dff[0],
                                        FLAGS.bind_enc_atv_fun[0], FLAGS.bind_fc_depth[0],
                                        FLAGS.bind_fc_units, FLAGS.bind_fc_atv_fun[0],
                                        FLAGS.affinity_enc_depth[0],
                                        FLAGS.affinity_enc_heads[0], FLAGS.affinity_enc_dff[0],
                                        FLAGS.affinity_enc_atv_fun[0], FLAGS.affinity_fc_depth[0],
                                        FLAGS.affinity_fc_units, FLAGS.affinity_fc_atv_fun[0])

    # SMILES Transformer-Encoder Optimizer
    smiles_opt = opt_config(FLAGS.smiles_bert_opt[0])
    # Binding Vector Prediction Optimizer
    bind_opt = opt_config(FLAGS.binding_bert_opt[0])
    # Binding Affinity Prediction Optimizer
    aff_opt = opt_config(FLAGS.affinity_bert_opt[0])

    # Binding Vector Final Dense Layer Initial Bias (Imbalanced Data Correction) np.log(pos/neg)
    # initial_bias = np.array([-2.4664805])
    initial_bias = np.array([-1.663877])

    model.get_layer('binding_output_block').get_layer('mlp_out').set_weights(
        [model.get_layer('binding_output_block').get_layer('mlp_out').get_weights()[0]] + [initial_bias])

    # Checkpoint Object and Checkpoint Manager
    global_var = tf.Variable(1)
    ckpt_obj = tf.train.Checkpoint(step=global_var, smiles_bert_opt=smiles_opt, bind_opt=bind_opt,
                                   affinity_opt=aff_opt, model=model)

    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt_obj, directory=FLAGS.checkpoint_dir,
                                              max_to_keep=1)

    # Model Summary
    model.summary()

    logger(("Prot Emb Size: %d, BERT SMILES Train: %d, Prot Enc Depth: %d, Prot Enc Heads: %d, Prot Enc DFF: "
            "%d, " +
            "Prot Enc AF: %s, Drop Rate: %0.4f, SMILES Pooler Dense Opt: %d, SMILES Pooler AF: %s, Bind Enc Depth: %d, "
            +
            "Bind Enc Heads: %d, Bind Enc DFF: %d, Bind Enc AF: %s, Bind FC Depth: %d, Bind FC Units: %s, "
            "Bind FC AF: %s, " +
            "Affinity Enc Depth: %d, Affinity Enc Heads: %d, Affinity Enc DFF: %d, Affinity Enc AF: %s, "
            "Affinity FC Depth: %d, " +
            "Affinity FC Units: %s, Affinity FC AF: %s, Batch: %d, Epochs: %d, Bind Pre-Train Epochs: %d, "
            "Bind Epochs Ratio: %d, " +
            "Affinity Epochs Ratio: %d, SMILES Bert Optimizer: %s, Bind Vector Optimizer: %s, Affinity "
            "Optimizer: %s, Bind Loss Opt: %s, " +
            "Fold: %d") %
           (FLAGS.prot_emb_size[0], FLAGS.bert_smiles_train[0], FLAGS.prot_enc_depth[0], FLAGS.prot_enc_heads[0],
            FLAGS.prot_enc_dff[0], FLAGS.prot_atv_fun[0], FLAGS.dropout_rate[0],  FLAGS.smiles_pooler_dense_opt[0],
            FLAGS.smiles_pooler_atv[0], FLAGS.bind_enc_depth[0], FLAGS.bind_enc_heads[0], FLAGS.bind_enc_dff[0],
            FLAGS.bind_enc_atv_fun[0], FLAGS.bind_fc_depth[0], FLAGS.bind_fc_units, FLAGS.bind_fc_atv_fun[0],
            FLAGS.affinity_enc_depth[0], FLAGS.affinity_enc_heads[0], FLAGS.affinity_enc_dff[0],
            FLAGS.affinity_enc_atv_fun[0], FLAGS.affinity_fc_depth[0], FLAGS.affinity_fc_units,
            FLAGS.affinity_fc_atv_fun[0], FLAGS.batch_size[0], FLAGS.epoch_num[0], FLAGS.pre_train_epochs[0],
            FLAGS.bind_vector_epochs[0], FLAGS.bind_affinity_epochs[0], smiles_opt.get_config(), bind_opt.get_config(),
            aff_opt.get_config(), FLAGS.bind_loss_opt[0], 0), 0, FLAGS)

    # Run Train/Validation
    results = run_train_val(FLAGS, 0, FLAGS.epoch_num[0], bind_data_train, bind_data_val, affinity_data_train,
                            affinity_data_val,
                            model, smiles_opt, bind_opt, aff_opt, bind_loss_opt, bind_loss_fn, bind_train_loss,
                            bind_train_accuracy, bind_train_recall, bind_train_precision, bind_train_f1,
                            bind_train_mcc, bind_val_loss, bind_val_accuracy, bind_val_recall,
                            bind_val_precision,
                            bind_val_f1, bind_val_mcc, affinity_loss_fn, affinity_train_loss,
                            affinity_train_rmse,
                            affinity_train_ci, affinity_val_loss, affinity_val_rmse, affinity_val_ci,
                            ckpt_obj, ckpt_manager, FLAGS.pre_train_epochs[0], FLAGS.bind_vector_epochs[0],
                            FLAGS.bind_affinity_epochs[0])

    logger("--------------Test/Val Metrics--------------", 'test', FLAGS)
    logger(("Bind Vector Val Loss: %0.4f, Bind Vector Val ACC: %0.4f, Bind Vector Val Recall: %0.4f, " +
            "Bind Vector Val Precision: %0.4f, Bind Vector Val F1: %0.4f, Bind Vector Val MCC: %0.4f, " +
            "Affinity Val Loss: %0.4f, Affinity Val RMSE: %0.4f, Affinity Val CI: %0.4f") % results, 'test',
           FLAGS)

def run_eval_model(FLAGS):
    with tf.device('/cpu:0'):
        # Binding Affinity Dataset
        affinity_data_path = {'data': '../data/affinity_data/davis_dataset_processed.csv',
                              'prot_dict': '',
                              'smiles_dict': '../dictionary/smiles_chembl_dict.txt',
                              'clusters': glob.glob('../data/affinity_data/clusters/*'),
                              'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                           '../dictionary/subword_units_map_uniprot.csv'],
                              'smiles_bpe': ''}

        affinity_protein, affinity_smiles, affinity_kd = dataset_builder(affinity_data_path).transform_dataset(1, 0,
                                                                                                               'Sequence',
                                                                                                               'SMILES',
                                                                                                               'Kd',
                                                                                                               FLAGS.prot_len,
                                                                                                               0,
                                                                                                               0,
                                                                                                               FLAGS.smiles_len)

        affinity_protein = add_reg_token(affinity_protein, FLAGS.prot_dict_len)
        affinity_smiles = add_reg_token(affinity_smiles, FLAGS.smiles_dict_len)

        _, _, _, clusters, _, _, _, _ = dataset_builder(affinity_data_path).get_data()

        affinity_folds = [list(clusters[i][1].iloc[:, 0]) for i in range(len(clusters)) if clusters[i][0] == 'test'][0]
        affinity_data = [affinity_protein, affinity_smiles, affinity_kd]

        affinity_data_val = [tf.gather(i, affinity_folds) for i in affinity_data]

        # Binding Pocket Dataset
        bind_data_val = load_data(FLAGS, '../data/bind_data/coach_test/prot.tfrecords',
                                '../data/bind_data/coach_test/smiles.tfrecords',
                                '../data/bind_data/coach_test/target.tfrecords',
                                '../data/bind_data/coach_test/weights.tfrecords')

        bind_data_val = [i for i in bind_data_val.as_numpy_iterator()]
        bind_data_val_prot = tf.concat([i[0][None,:] for i in bind_data_val],axis=0)
        bind_data_val_smiles = tf.concat([i[1][None, :] for i in bind_data_val], axis=0)
        bind_data_val_target = tf.concat([i[2][None, :, :] for i in bind_data_val], axis=0)
        bind_data_val_weights = tf.concat([i[3][None, :, :] for i in bind_data_val], axis=0)

    model = build_bert_binding_affinity(FLAGS, 256, 1, 3, 4, 1024, 'gelu', 0.1, 1, 'gelu', 1, 4, 1024, 'gelu', 3,
                        [128, 64, 32], 'gelu', 1, 4, 1024, 'gelu', 3, [1536, 1536, 1536], 'gelu')

    model = load_saved_model(model)

    aff_preds = model([affinity_data_val[0], affinity_data_val[1]], training=False)[1]
    bind_preds = model([bind_data_val_prot, bind_data_val_smiles], training=False)[0]

    aff_metrics = 'MSE: %0.4f, RMSE: %0.4f, CI: %0.4f, r2: %0.4f, Spearman: %0.4f' % \
                      inference_metrics(affinity_data_val[-1][:, None], aff_preds)

    bind_metrics = 'Balanced Accuracy: %0.4f, Recall: %0.4f, Precision: %0.4f, F1-Score: %0.4f, MCC: %0.4f' % \
                       bind_metrics_function(bind_data_val_target, bind_preds, bind_data_val_weights)


    print('-----------Davis Independent Testing Set Metrics-----------')
    print(aff_metrics)
    pred_scatter_plot(affinity_data_val[-1][:, None], aff_preds,
                      'Davis Dataset: Predictions vs True Values', 'True Values',
                      'Predictions', False, '')

    print('-----------COACH Testing Set Metrics-----------')
    print(bind_metrics)


if __name__ == '__main__':
    FLAGS = parser_arg()

    if FLAGS.inference_option == 'Train':
        run_train_model(FLAGS)

    if FLAGS.inference_option == 'Validation':
        run_grid_search(FLAGS)

    if FLAGS.inference_option == 'Evaluation':
        run_eval_model(FLAGS)


