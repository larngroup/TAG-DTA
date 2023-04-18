# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import argparse
import os


def argparser():
    """
    Argument Parser Function

    Outputs:
    - FLAGS: arguments object

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=dict,
        default={},
        help='Data Path')

    parser.add_argument(
        '--pos_enc_option',
        type=bool,
        default=True,
        help='Position encoding option')

    parser.add_argument(
        '--smiles_len',
        type=int,
        default=100,
        help='SMILES Max Length')

    parser.add_argument(
        '--smiles_dict_len',
        type=int,
        default=72,
        help='SMILES Dictionary Length')

    parser.add_argument(
        '--dense_atv_fun',
        type=str,
        nargs='+',
        help='Dense Layer Activation Function')

    parser.add_argument(
        '--return_intermediate',
        type=bool,
        default=False,
        help='Return Intermediate Values (Enc,Cross)')

    parser.add_argument(
        '--transformer_depth',
        type=int,
        nargs='+',
        help='Transformer Encoder Depth')

    parser.add_argument(
        '--d_model',
        type=int,
        nargs='+',
        help='Emb Size')

    parser.add_argument(
        '--transformer_heads',
        type=int,
        nargs='+',
        help='Transformer Encoder Heads')


    parser.add_argument(
        '--d_ff_dim',
        type=int,
        nargs='+',
        help='PosWiseFF Dim')
        
    parser.add_argument(
        '--parameter_sharing',
        type=str,
        nargs='+',
        help='Parameter Sharing Option')

    parser.add_argument(
        '--dim_k',
        type=int,
        nargs='+',
        help='Projection Dimension (Dim K)')

    parser.add_argument(
        '--full_attn',
        type=bool,
        default=True,
        help='Full Attention Option')

    parser.add_argument(
        '--dropout_rate',
        type=float,
        nargs='+',
        help='Dropout Rate')

    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action='append',
        help='Optimizer Function Parameters')

    parser.add_argument(
        '--batch_dim',
        type=int,
        nargs='+',
        help='Batch Dim')

    parser.add_argument(
        '--num_epochs',
        type=int,
        nargs='+',
        help='Number of Epochs')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Directory for checkpoint weights'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Directory for log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    """
    Logging function to update the log file

    Args:
    - msg [str]: info to add to the log file
    - FLAGS: arguments object

    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")

    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")
