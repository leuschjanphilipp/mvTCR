from comet_ml import Experiment
import scanpy as sc
import os
import random
import torch
import numpy as np

from tcr_embedding.models.mixture_modules.rna_model import RnaModel
from tcr_embedding.models.mixture_modules.separate_model import SeparateModel
from tcr_embedding.models.mixture_modules.poe import PoEModel
from tcr_embedding.models.mixture_modules.moe import MoEModel


def fix_seeds(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_data(source='10x'):
    """
    Loads the whole dataset from a defined source.
    :param source: str indicting the dataset or filename starting from the data_folder
    :return: adata object
    """
    path_current = os.path.dirname(__file__)
    path_base = os.path.join(path_current, '../data/')

    source = source.lower()
    if source == '10x':
        path_source = '10x_CD8TC/v6_supervised.h5ad'
    elif source == 'bcc':
        path_source = 'BCC/06_bcc_highly_var_5000.h5ad'
    elif source == 'scc':
        path_source = 'SCC/06_scc_highly_var_5000.h5ad'
    elif source == 'covid':
        path_source = 'Covid/04_covid_highly_var_5000.h5ad'
    elif source == 'haniffa':
        path_source = 'Haniffa/v3_conditional.h5ad'
    else:
        path_source = source
    path_file = os.path.join(path_base, path_source)

    try:
        data = sc.read_h5ad(path_file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Data file not found at {path_file}. '
                                f'Please specify data source by "10x", "bcc", "scc", "haniffa", or "covid". '
                                f'Alternatively, as filename starting from the data folder.')
    return data


def load_model(adata, path_model):
    path_base = os.path.dirname(__file__)
    path_model = os.path.join(path_base, '..', path_model)
    model_file = torch.load(path_model)

    params_architecture = model_file['params_architecture']
    balanced_sampling = model_file['balanced_sampling']
    metadata = model_file['metadata']
    conditional = model_file['conditional']
    optimization_mode_params = model_file['optimization_mode_params']
    label_key = model_file['label_key']

    model_class = select_model_by_name(model_file['model_type'])
    model = model_class(adata, params_architecture, balanced_sampling, metadata, conditional,
                        optimization_mode_params, label_key)
    model.load(path_model)
    return model


def initialize_comet(params_hpo, params_fixed):
    if 'comet' not in params_fixed or params_fixed['comet'] is None:
        return None

    experiment_name = params_fixed['name']

    path_key = os.path.join(os.path.dirname(__file__), '../config/API_key.txt')
    with open(path_key) as f:
        comet_key = f.read()
    experiment = Experiment(api_key=comet_key, workspace=params_fixed['workspace'], project_name=experiment_name)

    experiment.log_parameters(params_hpo)
    experiment.log_parameters(params_hpo['scRNA_model_hyperparams'], prefix='scRNA')
    experiment.log_parameters(params_hpo['seq_model_hyperparams'], prefix='seq')

    experiment.log_parameters(params_fixed, prefix='fixed_params')

    if params_hpo['seq_model_arch'] == 'CNN':
        experiment.log_parameters(params_hpo['seq_model_hyperparams']['encoder'], prefix='seq_encoder')
        experiment.log_parameters(params_hpo['seq_model_hyperparams']['decoder'], prefix='seq_decoder')
    return experiment


def select_model_by_name(model_name):
    """
    Select between modeltypes (e.g. single, concat, poe, moe, ...) by an identifier
    :param model_name:  str indicating the model type, type is chosen, when following indicator are part of the string
                        default: joint model
                        'seperate': for individual alpha beta chain representation
                        'moe': mixture of experts
                        'poe': product of experts
    :return: class of the corresponding model
    """
    init_dict = {
        'rna': RnaModel,
        'separate': SeparateModel,
        'moe': MoEModel,
        'poe': PoEModel,
    }
    model_name = model_name.lower()
    if model_name in init_dict:
        return init_dict[model_name]
    else:
        raise ValueError('Please specify a valid model name')


def determine_marker_genes(adata, resolution, visualize=False, filter_tcr=True):
    adata = adata[adata.obs['set'] == 'train']
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)
    if visualize:
        sc.tl.umap(adata)  # for visualization only
        sc.pl.umap(adata, color='leiden')

    # Filter TRA, TRB and TRG which forms the T-cell receptor
    if filter_tcr:
        adata = adata[:, ~adata.var.index.str.contains('TRA')]
        adata = adata[:, ~adata.var.index.str.contains('TRB')]
        adata = adata[:, ~adata.var.index.str.contains('TRG')]

    sc.tl.rank_genes_groups(adata, groupby='leiden')
    highly_variable = list(adata.uns['rank_genes_groups']['names'][0])
    return highly_variable


def plot_umap_list(adata, title, color_groups):
    """
    Plots UMAPS based with different coloring groups
    :param adata: Adata Object containing a latent space embedding
    :param title: Figure title
    :param color_groups: Column name in adata.obs used for coloring the UMAP
    :return:
    """
    try:
        if adata.X.shape[1] == 2:
            adata.obsm['X_umap'] = adata.X
        else:
            sc.pp.neighbors(adata, use_rep='X')
            sc.tl.umap(adata)
        figures = []
        for group in color_groups:
            fig = sc.pl.umap(adata, color=group, title=title+'_'+group, return_fig=True)
            fig.tight_layout()
            figures.append(fig)
        return figures
    except ValueError as e:
        print(e)
        return []
