"""
python -u borcherding_optuna.py --model moe
"""
# comet-ml must be imported before torch and sklearn
import comet_ml

import sys
sys.path.append('..')

import numpy as np
from tcr_embedding.models.model_selection import run_model_selection
import tcr_embedding.utils_training as utils
from tcr_embedding.utils_preprocessing import group_shuffle_split

import os
import argparse


random_seed = 42
utils.fix_seeds(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--rna_weight', type=int, default=1)
parser.add_argument('--model', type=str, default='moe')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--wo_tcr_genes', action='store_true')
parser.add_argument('--conditional', type=str, default=None)

args = parser.parse_args()


adata = utils.load_data('borcherding')

if args.wo_tcr_genes:
    tcr_gene_prefixs = ['TRAV', 'TRAJ', 'TRAC', 'TRB', 'TRDV', 'TRDC', 'TRG']
    non_tcr_genes = adata.var_names
    for prefix in tcr_gene_prefixs:
        non_tcr_genes = [el for el in non_tcr_genes if not el.startswith(prefix)]
    adata = adata[:, non_tcr_genes]

# Randomly select patients to be left out during training
def get_n_patients(amount_patients):
    if amount_patients <= 5:
        return 0
    else:
        return 2


holdout_patients = {}

adata.obs['Tissue+Type'] = [f'{tissue}.{type_}' for tissue, type_ in zip(adata.obs['Tissue'], adata.obs['Type'])]
counts = adata.obs.groupby('Tissue+Type')['Sample'].value_counts()
for cat in adata.obs['Tissue+Type'].unique():
    n = get_n_patients(len(counts[cat]))
    choice = np.random.choice(counts[cat].index, n, replace=False).tolist()
    holdout_patients[cat] = choice

for patients in holdout_patients.values():
    adata = adata[~adata.obs['Sample'].isin(patients)]

train, val = group_shuffle_split(adata, group_col='clonotype', val_split=0.2, random_seed=random_seed)
adata.obs['set'] = 'train'
adata.obs.loc[val.obs.index, 'set'] = 'val'
adata = adata[adata.obs['set'].isin(['train', 'val'])]


params_experiment = {
    'study_name': f'borcherding_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}',
    'comet_workspace': None,  # 'Covid',
    'model_name': args.model,
    'balanced_sampling': 'clonotype',
    'metadata': ['clonotype', 'Sample', 'Type', 'Tissue', 'Tissue+Type', 'functional.cluster'],
    'save_path': os.path.join(os.path.dirname(__file__), '..', 'optuna', f'borcherding_{args.model}_{args.rna_weight}_{args.conditional}_{args.wo_tcr_genes}'),
    'conditional': args.conditional
}
if args.model == 'rna':
    params_experiment['balanced_sampling'] = None

params_optimization = {
    'name': 'pseudo_metric',
    'prediction_labels':
        {'clonotype': 1,
         'Tissue+Type': args.rna_weight}
}

timeout = (2 * 24 * 60 * 60) - 300
run_model_selection(adata, params_experiment, params_optimization, None, timeout, args.gpus)
