from argument_parser import *
from dataset_builder_util import *


if __name__ == '__main__':
    FLAGS = argparser()
    FLAGS.data_path = {'data': '../data/chembl_smiles_proc.tsv',
                       'smiles_dict': '../dictionary/smiles_chembl_dict.txt'}


    # Save ChEMBL dataset in TFRecord format
    save_proc_data(FLAGS)

    # Save ChEBML Dataset for MLM Pre-Train in TFRecord formats
    save_data_mask(FLAGS)


    
