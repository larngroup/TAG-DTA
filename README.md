# BR-DTATR: End-to-End Binding Region-Guided Strategy to Predict Drug-Target Affinity Using Transformers
<p align="justify"> We propose an end-to-end binding region-guided Transformer-based architecture that simultaneously predicts the 1D binding pocket and the binding affinity of DTI pairs, where the prediction of the 1D binding pocket guides and conditions the prediction of DTA. This architecture uses 1D raw sequential and structural data to represent the proteins and compounds, respectively, and combines multiple Transformer-Encoder blocks to capture and learn the proteomics, chemical, and pharmacological contexts. The predicted 1D binding pocket conditions the attention mechanism of the Transformer-Encoder used to learn the pharmacological space in order to model the inter-dependency amongst binding-related positions.</p>


## BR-DTATR Architecture
<p align="center"><img src="/figures/br_dtatr.png" width="90%" height="90%"/></p>

## DTI/Model Explainability
### ABL1(E255K)-phosphorylated - SKI-606
<p align="center"><img src="/figures/docking_br_dtatr_v1_a.png" width="90%" height="90%"/></p>
<p align="center"><img src="/figures/docking_br_dtatr_v1_b.png" width="90%" height="90%"/></p>

## Data Availability
## Binding Affinity Data
### Dataset
- **davis_dataset_processed:** Davis Dataset Processed : prot sequences + rdkit SMILES strings + pkd values
### Clusters
- **test_cluster:** independent test set indices
- **train_cluster_X:** train indices 

## Binding Pocket Data
### Datasets
- **scPDB + PDBBind + BioLip:** Training/Validation Binding Pocket Dataset (TFRecords Format)
- **Coach Test:** Testing Binding Pocket Dataset (TFRecords Format)

## SMILES Pre-Train MLM
### Datasets
- **ChEMBL Dataset**: Training/Validation SMILES Dataset (TFRecords Format)

## Dictionaries
- **smiles_chembl_dict**: SMILES char-integer dictionary
- **protein_codes_uniprot/subword_units_map_uniprot**: Protein Subwords Dictionary

## Requirements:
- Python 3.9.6
- Tensorflow 2.8.0
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Glob
- Json
- periodictable
- subword_nmt

## Usage 
(The architecture supports the use of the Linear Multi-Head Attention arXiv:2006.04768)
## BR-DTATR (Dir:'./BR-DTATR/source/')
### Training
```
python main.py --inference_option Train --prot_emb_size 256 --bert_smiles_train 1 --prot_enc_depth 3 --prot_enc_heads 4 --prot_enc_dff 1024 --prot_atv_fun gelu --dropout_rate 0.1 --smiles_pooler_dense_opt 1 --smiles_pooler_atv gelu --bind_enc_depth 1 --bind_enc_heads 4 --bind_enc_dff 1024 --bind_enc_atv_fun gelu --bind_fc_depth 3 --bind_fc_units 128 64 32 --bind_fc_atv_fun gelu --affinity_enc_depth 1 --affinity_enc_heads 4 --affinity_enc_dff 1024 --affinity_enc_atv_fun gelu --affinity_fc_depth 3 --affinity_fc_units 1536 1536 1536 --affinity_fc_atv_fun gelu --batch_size 32 --epoch_num 500 --pre_train_epochs 20 --bind_vector_epochs 1 --bind_affinity_epochs 3 --smiles_bert_opt radam 1e-05 0.9 0.999 1e-08 1e-05 0 0 0 --binding_bert_opt radam 1e-04 0.9 0.999 1e-08 1e-05 0 0 0 --affinity_bert_opt radam 1e-04 0.9 0.999 1e-08 1e-05 0 0 0 --bind_loss_opt standard 0.40 0.60
```
### Validation
```
python main.py --inference_option Validation --prot_emb_dff 256 1024 --bert_smiles_train 1 --prot_enc_depth 3 --prot_enc_heads 4 --prot_atv_fun gelu --dropout_rate 0.1 --smiles_pooler_dense_opt 1 --smiles_pooler_atv gelu --bind_enc_depth 1 --bind_enc_heads 4 --bind_enc_atv_fun gelu --bind_fc_depth 3 --bind_fc_units 128 64 32 --bind_fc_atv_fun gelu --affinity_enc_depth 1 --affinity_enc_heads 4 --affinity_enc_atv_fun gelu --affinity_fc_depth 3 --affinity_fc_units 384 192 96 --affinity_fc_atv_fun gelu --batch_size 32 --epoch_num 500 --pre_train_epochs 20 --bind_vector_epochs 1 --bind_affinity_epochs 3 --smiles_bert_opt radam 1e-05 0.9 0.999 1e-08 1e-05 0 0 0 --binding_bert_opt radam 1e-04 0.9 0.999 1e-08 1e-05 0 0 0 --affinity_bert_opt radam 1e-04 0.9 0.999 1e-08 1e-05 0 0 0 --bind_loss_opt standard 0.40 0.60
```

### Evaluation
```
python main.py --inference_option Evaluation
```

## SMILES Pre-Train MLM (Dir:'./SMILES-MLM/source/')
```
python bert_mlm.py --num_epochs 500 --batch_dim 246 --transformer_depth 3 --transformer_heads 8 --d_ff_dim 2048 --d_model 512 --dropout_rate 0.1 --dense_atv_fun gelu --optimizer_fn radam 1e-03 0.9 0.999 1e-08 1e-04 --dim_k 0 --parameter_sharing ''
```