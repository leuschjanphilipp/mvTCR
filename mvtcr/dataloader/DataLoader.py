import random
import torch
import numpy as np
import pandas as pd
import scirpy as ir

from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from mvtcr.dataloader.Dataset import JointDataset


def create_datasets(adata, obs_set_key, tcr_chain, use_vdj, use_citeseq,                    
                    metadata=None, conditional=None, labels=None):
    """
    Create torch Dataset, see above for the input
    :param adata: adata
    :param val_split:
    :param metadata:
    :param conditional:
    :param labels:
    :return: train_dataset, val_dataset, train_masks (for continuing training)
    """

    # Splits everything into train and val
    if obs_set_key is not None:
        train_mask = (adata.obs[obs_set_key] == 'train').values
    else:
        train_mask = np.array([True] * adata.obs.shape[0])
    
    if tcr_chain == "alpha":
        tcr_seq = adata.obsm['alpha_seq']
        tcr_length = np.column_stack([adata.obs['alpha_len']])
    elif tcr_chain == "beta":
        tcr_seq = adata.obsm['beta_seq']
        tcr_length = np.column_stack([adata.obs['beta_len']])
    elif tcr_chain == "both":
        tcr_seq = np.concatenate([adata.obsm['alpha_seq'], adata.obsm['beta_seq']], axis=1)
        tcr_length = np.column_stack([adata.obs['alpha_len'], adata.obs['beta_len']])
    else:
        print("tcr_chain mode not supported. Chose one of {'alpha', 'beta', 'both'}.")

    tcr_train = tcr_seq[train_mask]
    tcr_val = tcr_seq[~train_mask]

    tcr_length_train = tcr_length[train_mask].tolist()
    tcr_length_val = tcr_length[~train_mask].tolist()

    # Save dataset splits
    rna_train = adata.X[train_mask]
    rna_val = adata.X[~train_mask]

    if use_vdj:
        vdj_data = ir.get.airr(adata, ["v_call", "d_call", "j_call"]).copy()
        #vdj_data = pd.get_dummies(vdj_data, columns=["VJ_1_v_call", "VJ_1_j_call", "VDJ_1_v_call", "VDJ_1_d_call", "VDJ_1_j_call"], dtype=int).to_numpy()
        for col in vdj_data.columns:
            unique_values = vdj_data[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            vdj_data[col] = vdj_data[col].map(mapping)
        vdj_data = vdj_data.to_numpy() #np array 5 genes x cells
        
        vdj_train = vdj_data[train_mask]
        vdj_val = vdj_data[~train_mask]
    else:
        vdj_train, vdj_val = None, None
    
    if use_citeseq:
        #TODO haniffa, normalize with clr
        citeseq_train, citeseq_val = None, None
    else:
        citeseq_train, citeseq_val = None, None

    if metadata is None:
        metadata = []
    
    metadata_train = adata.obs[metadata][train_mask].to_numpy()
    metadata_val = adata.obs[metadata][~train_mask].to_numpy()

    if labels is not None:
        labels = adata.obs[labels].to_numpy()
        labels_train = labels[train_mask]
        labels_val = labels[~train_mask]
    else:
        labels_train, labels_val = None, None

    if conditional is not None:
        conditional_train = adata.obsm[conditional][train_mask]
        conditional_val = adata.obsm[conditional][~train_mask]
    else:
        conditional_train, conditional_val = None, None


    train_dataset = JointDataset(tcr_train, tcr_length_train, rna_train, vdj_train, citeseq_train, metadata_train,
                                 labels_train, conditional_train)
    val_dataset = JointDataset(tcr_val, tcr_length_val, rna_val, vdj_val, citeseq_val, metadata_val,
                                 labels_val, conditional_val)
    #TODO why train mask return? --> sampling weights
    return train_dataset, val_dataset, train_mask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# <- functions for the main data loader ->
def initialize_data_loader(adata, obs_set_key, tcr_chain, use_vdj, use_citeseq, metadata, conditional, label_key, balanced_sampling, batch_size):
    
    train_datasets, val_datasets, train_mask = create_datasets(adata, 
                                                               obs_set_key=obs_set_key,
                                                               tcr_chain=tcr_chain,
                                                               use_vdj=use_vdj, 
                                                               use_citeseq=use_citeseq, 
                                                               metadata=metadata, 
                                                               conditional=conditional, 
                                                               labels=label_key)

    if balanced_sampling is None:
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
    else:
        sampling_weights = calculate_sampling_weights(adata, train_mask, class_column=balanced_sampling)
        sampler = WeightedRandomSampler(weights=sampling_weights, num_samples=len(sampling_weights),
                                        replacement=True)
        # shuffle is mutually exclusive to sampler, but sampler is anyway shuffled
        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False,
                                  sampler=sampler, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def calculate_sampling_weights(adata, train_mask, class_column):
    """
    Calculate sampling weights for more balanced sampling in case of imbalanced classes,
    :params class_column: str, key for class to be balanced
    :params log_divisor: divide the label counts by this factor before taking the log, higher number makes the sampling more uniformly balanced
    :return: list of weights
    """
    label_counts = []

    label_count = adata[train_mask].obs[class_column].map(adata[train_mask].obs[class_column].value_counts())
    label_counts.append(label_count)

    label_counts = pd.concat(label_counts, ignore_index=True)
    label_counts = np.log(label_counts / 10 + 1)
    label_counts = 1 / label_counts

    sampling_weights = label_counts / sum(label_counts)
    return sampling_weights


# <- data loader for prediction ->
def initialize_prediction_loader(adata, tcr_chain, use_vdj, use_citeseq, metadata, batch_size, conditional=None):
    prediction_dataset, _, _ = create_datasets(adata, 
                                               obs_set_key=None, 
                                               tcr_chain=tcr_chain,
                                               use_vdj=use_vdj, 
                                               use_citeseq=use_citeseq, 
                                               metadata=metadata, 
                                               conditional=conditional,
                                               labels=None)
    
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)
    return prediction_loader


# <- data loader for calculating the transcriptome from the latent space ->
def initialize_latent_loader(adata_latent, batch_size, conditional):
    if conditional is None:
        dataset = TensorDataset(torch.from_numpy(adata_latent.X))
    else:
        dataset = TensorDataset(torch.from_numpy(adata_latent.X),
                                                 torch.from_numpy(adata_latent.obsm[conditional]))
    latent_loader = DataLoader(dataset, batch_size=batch_size)
    return latent_loader