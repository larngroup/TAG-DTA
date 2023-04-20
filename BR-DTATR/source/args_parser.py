import argparse
import os


def parser_arg():
    """
    Argument Parser Function

    Outputs:
    - FLAGS: arguments object

    """

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--inference_option',
        type=str,
        help='Train, Validation, or Evaluation'
    )

    parser.add_argument(
        '--bert_smiles_model_dict',
        type=list,
        default=(100, 72, True, False, 512, 0.1, 3, 8, 2048, 'gelu', 0, ""),
        help='Pre-Trained SMILES Transformer-Encoder Parameters'
    )

    parser.add_argument(
        '--bert_smiles_model_ckpt_obj',
        type=str,
        default='../smiles_mlm_checkpoint/',
        help='Pre-Trained SMILES Transformer-Encoder Checkpoint Path'
    )

    parser.add_argument(
        '--bert_smiles_train',
        type=int,
        nargs='+',
        help='Pre-Trained SMILES Transformer-Encoder Training Option: 1 - Train, 0 - Freeze weights'
    )

    parser.add_argument(
        '--prot_len',
        default=575,
        type=int,
        help='Protein Sequence Maximum Length'
    )

    parser.add_argument(
        '--smiles_len',
        default=100,
        type=int,
        help='SMILES String Maximum Length'
    )

    parser.add_argument(
        '--smiles_emb_size',
        default=512,
        type=int,
        help='SMILES Strings Embedding size'
    )

    parser.add_argument(
        '--prot_dict_len',
        default=16693,
        type=int,
        help='Protein Dictionary Cardinality (Number of unique tokens excluding special tokens)'
    )

    parser.add_argument(
        '--smiles_dict_len',
        default=72,
        type=int,
        help='SMILES Dictionary Cardinality (Number of unique tokens excluding special tokens)'
    )


    parser.add_argument(
        '--prot_emb_dff',
        type=int,
        nargs='+',
        action='append',
        help='Protein Embedding Size & Protein Encoder Hidden Dimension'
    )

    parser.add_argument(
        '--prot_emb_size',
        type=int,
        nargs='+',
        help='Protein Embedding Size'
    )

    parser.add_argument(
        '--prot_enc_depth',
        type=int,
        nargs='+',
        help='Number of Protein Transformer-Encoder Layers'
    )

    parser.add_argument(
        '--prot_enc_heads',
        type=int,
        nargs='+',
        help='Number of Protein Transformer-Encoder Heads of Attention'
    )

    parser.add_argument(
        '--prot_enc_dff',
        type=int,
        nargs='+',
        help='Protein Transformer-Encoder PWFFN Expansion Ratio (Number of hidden units for the first dense layer)'
    )

    parser.add_argument(
        '--prot_atv_fun',
        type=str,
        nargs='+',
        help='Protein Transformer-Encoder Activation Function'
    )

    parser.add_argument(
        '--prot_enc_dim_k',
        type=int,
        default=0,
        help='Protein Transformer-Encoder MHA Linear Attention Dim K'
    )

    parser.add_argument(
        '--prot_enc_param_share',
        type=str,
        default='',
        help='Protein Transformer-Encoder MHA Linear Attention Param Sharing Option: : "layerwise", "none", "headwise" '
    )

    parser.add_argument(
        '--prot_enc_full_attention',
        type=int,
        default=1,
        help='Protein Transformer-Encoder MHA Attention Model: 1 - Full Attention, 0 - Linear Attention'

    )

    parser.add_argument(
        '--prot_enc_return_interm',
        type=int,
        default=0,
        help='Protein Transformer-Encoder Return Intermediate Values'

    )

    parser.add_argument(
        '--smiles_pooler_dense_opt',
        type=int,
        nargs='+',
        help='SMILES Transformer-Encoder PWPool Block Projection Option: 1 - Project last dimension'
    )

    parser.add_argument(
        '--smiles_pooler_atv',
        type=str,
        nargs='+',
        help='SMILES Transformer-Encoder PWPool Block Activation Function'
    )

    parser.add_argument(
        '--bind_enc_depth',
        type=int,
        nargs='+',
        help='Number of Binding Pocket Transformer-Encoder Layers'
    )

    parser.add_argument(
        '--bind_enc_heads',
        type=int,
        nargs='+',
        help='Number of Binding Pocket Transformer-Encoder Heads of Attention'
    )

    parser.add_argument(
        '--bind_enc_dff',
        type=int,
        nargs='+',
        help='Binding Pocket Transformer-Encoder PWFFN Expansion Ratio (Number of hidden units for the first dense layer)'
    )

    parser.add_argument(
        '--bind_enc_atv_fun',
        type=str,
        nargs='+',
        help='Binding Pocket Transformer-Encoder Activation Function'
    )

    parser.add_argument(
        '--bind_enc_dim_k',
        type=int,
        default=0,
        help='Binding Pocket Transformer-Encoder MHA Linear Attention Dim K'
    )

    parser.add_argument(
        '--bind_enc_param_share',
        type=str,
        default='',
        help='Binding Pocket Transformer-Encoder MHA Linear Attention Param Sharing Option: : "layerwise", "none", "headwise" '
    )

    parser.add_argument(
        '--bind_enc_full_attention',
        type=int,
        default=1,
        help='Binding Pocket Transformer-Encoder MHA Attention Model: 1 - Full Attention, 0 - Linear Attention'

    )

    parser.add_argument(
        '--bind_enc_return_interm',
        type=int,
        default=0,
        help='Binding Pocket Transformer-Encoder Return Intermediate Values'

    )

    # parser.add_argument(
    #     '--bind_fc_depth_units',
    #     type=int,
    #     nargs='+',
    #     help='Number of FC Dense Layers and Hidden Units'
    # )

    parser.add_argument(
        '--bind_fc_depth',
        type=int,
        nargs='+',
        help='Binding Pocket Classifier: Number of PWMLP Dense Layers'
    )

    parser.add_argument(
        '--bind_fc_units',
        type=int,
        nargs='+',
        # action='append',
        help='Binding Pocket Classifier: PWMLP Hidden Units'

    )

    parser.add_argument(
        '--bind_fc_atv_fun',
        type=str,
        nargs='+',
        help='Binding Pocket Classifier: PWMLP Activation Function'
    )


    parser.add_argument(
        '--affinity_enc_depth',
        type=int,
        nargs='+',
        help='Number of Binding Affinity Transformer-Encoder Layers'
    )

    parser.add_argument(
        '--affinity_enc_heads',
        type=int,
        nargs='+',
        help='Number of Binding Affinity Transformer-Encoder Heads of Attention'
    )

    parser.add_argument(
        '--affinity_enc_dff',
        type=int,
        nargs='+',
        help='Binding Affinity Transformer-Encoder PWFFN Expansion Ratio (Number of hidden units for the first dense layer)'
    )

    parser.add_argument(
        '--affinity_enc_atv_fun',
        type=str,
        nargs='+',
        help='Binding Affinity Transformer-Encoder Activation Function'
    )

    parser.add_argument(
        '--affinity_enc_dim_k',
        type=int,
        default=0,
        help='Binding Affinity Transformer-Encoder MHA Linear Attention Dim K'
    )

    parser.add_argument(
        '--affinity_enc_param_share',
        type=str,
        default='',
        help='Binding Affinity Transformer-Encoder MHA Linear Attention Param Sharing Option: : "layerwise", "none", "headwise" '
    )

    parser.add_argument(
        '--affinity_enc_full_attention',
        type=int,
        default=1,
        help='Binding Affinity Transformer-Encoder MHA Attention Model: 1 - Full Attention, 0 - Linear Attention'

    )

    parser.add_argument(
        '--affinity_enc_return_interm',
        type=int,
        default=0,
        help='Binding Affinity Transformer-Encoder Return Intermediate Values'

    )

    # parser.add_argument(
    #     '--affinity_fc_depth_units',
    #     type=int,
    #     nargs='+',
    #     help='Number of FC Dense Layers and Hidden Units'
    # )

    parser.add_argument(
        '--affinity_fc_depth',
        type=int,
        nargs='+',
        help='Binding Affinity Regressor: Number of FCNN Dense Layers'
    )

    parser.add_argument(
        '--affinity_fc_units',
        type=int,
        nargs='+',
        # action='append',
        help='Binding Affinity Regressor: FCNN Hidden Units'

    )

    parser.add_argument(
        '--affinity_fc_atv_fun',
        type=str,
        nargs='+',
        help='Binding Affinity Regressor: FCNN  Activation Function'
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        nargs='+',
        help='Dropout Rate'
    )

    # parser.add_argument(
    #     '--output_act_fun',
    #     type=str,
    #     default='linear',
    #     help='Output Dense Layer Activation Function'
    # )

    parser.add_argument(
        '--batch_size',
        type=int,
        nargs='+',
        help='Batch Size'
    )

    parser.add_argument(
        '--epoch_num',
        type=int,
        nargs='+',
        help='Number of epochs'
    )

    parser.add_argument(
        '--pre_train_epochs',
        type=int,
        nargs='+',
        help='Binding Vector Pre-Training Epochs'
    )

    parser.add_argument(
        '--bind_vector_epochs',
        type=int,
        nargs='+',
        help='Binding Pocket Model Epochs Ratio'
    )

    parser.add_argument(
        '--bind_affinity_epochs',
        type=int,
        nargs='+',
        help='Binding Affinity Model Epochs Ratio'
    )

    parser.add_argument(
        '--smiles_bert_opt',
        type=str,
        nargs='+',
        action='append',
        help='Pre-Trained SMILES Transformer-Encoder Optimizer Function'
    )

    parser.add_argument(
        '--binding_bert_opt',
        type=str,
        nargs='+',
        action='append',
        help='Binding Pocket Classifier Optimizer Function'
    )

    parser.add_argument(
        '--bind_loss_opt',
        type=str,
        nargs='+',
        action='append',
        help='Binding Pocket Loss Function Option'
    )

    parser.add_argument(
        '--affinity_bert_opt',
        type=str,
        nargs='+',
        action='append',
        help='Binding Affinity Regressor Optimizer Function'
    )


    parser.add_argument(
        '--affinity_drop_lr_train_cycle',
        type=int,
        default=25,
        help='Binding Affinity Regressor Optimizer: Number of Training Cycles that the LR keeps constant'
    )

    parser.add_argument(
        '--binding_vector_drop_lr_train_cycle',
        type=int,
        default=25,
        help='Binding Pocket Classifier Optimizer: Number of Training Cycles that the LR keeps constant'
    )

    parser.add_argument(
        '--es_train_cycle',
        type=int,
        default=10,
        help='Early Stopping Patience based on Training Cycles'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Logging Directory'
    )

    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='',
        help='Checkpoints Directory'
    )


    FLAGS, _ = parser.parse_known_args()
    return FLAGS


def logger(msg, num, FLAGS):
    """
    Logging function to update the log file

    Args:
    - msg [str]: info to add to the log file
    - num [int]: fold number
    - FLAGS: arguments object

    """
    
    fpath = os.path.join(FLAGS.log_dir, 'log_{}.txt'.format(str(num)))

    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")
