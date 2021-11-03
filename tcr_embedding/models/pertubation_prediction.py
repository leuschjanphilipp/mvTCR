from tcr_embedding.evaluation.PertubationPrediction import evaluate_pertubation
import numpy as np
import scanpy as sc

sc.settings.verbosity = 0


def predict_pertubation(latent_train, latent_val, model, column_perturbation, indicator_perturbation, var_names):
    """
    Predict the effect of pertubation on transcriptome level
    :param latent_train: adata object containing the latent spaces of the training dataset
    :param latent_val: adata object containing the latent spaces of the validaiton dataset
    :param model: model to predict transcritome from latent space
    :param column_perturbation: str, column in the adata objects indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the perturbation
    :param var_names: list containing the gene names
    :return: adata object, containing the predicted transcriptome profile after perturbation
    """
    # todo delta per cell type?
    delta = get_delta(latent_train, column_perturbation, indicator_perturbation)
    adata_pred = sc.AnnData(latent_val.X + delta)
    adata_pred = model.predict_transcriptome_from_latent(adata_pred)
    adata_pred.var_names = var_names
    return adata_pred


def get_delta(adata_latent, column_perturbation, indicator_perturbation):
    """
    Calculate the difference vector between pre and post perturbation in the latent space
    :param adata_latent: adata oject, containing the latent space representation of the training data
    :param column_perturbation: str, column in the adata object indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the profile after perturbation
    :return: numpy array, difference vectore
    """
    mask_pre = adata_latent.obs[column_perturbation] == indicator_perturbation
    latent_pre = adata_latent[mask_pre]
    latent_post = adata_latent[~mask_pre]
    avg_pre = np.mean(latent_pre.X, axis=0)
    avg_post = np.mean(latent_post.X, axis=0)
    delta = avg_post - avg_pre
    return delta


def run_scgen_cross_validation(adata, column_fold, model, column_perturbation, indicator_perturbation, column_cluster):
    """
    Runs perturbation prediction over a specified fold column and evaluates the results
    :param adata: adata object, of the raw data
    :param column_fold: str, indicating the column over which to cross validate
    :param model: model used for latent space generation
    :param column_perturbation: str, column in the adata object indicating the pertubation
    :param indicator_perturbation: str, value for the 'pre' state of the profile after perturbation
    :param column_cluster: column indicating the clusters for which the top 100 DEG are determined
    :return: dict, summary over performance on the different splits and aggregation
    """
    latent_full = model.get_latent(adata, metadata=[column_fold, column_perturbation])

    summary_performance = {}
    rs_squared = []
    for fold in adata.obs[column_fold].unique():
        mask_train = latent_full.obs[column_fold] != fold
        latent_train = latent_full[mask_train]
        latent_val = latent_full[~mask_train]

        if not 0 < sum(latent_train.obs[column_perturbation]==indicator_perturbation) < len(latent_train):
            continue
        if not 0 < sum(latent_val.obs[column_perturbation]==indicator_perturbation) < len(latent_val):
            continue

        mask_val_pre = latent_val.obs[column_perturbation] == indicator_perturbation
        latent_val_pre = latent_val[mask_val_pre]

        pred_val_post = predict_pertubation(latent_train, latent_val_pre, model,
                                            column_perturbation, indicator_perturbation,
                                            var_names=adata.var_names)
        score = evaluate_pertubation(adata[adata.obs[column_fold] == fold].copy(), pred_val_post, None,
                                     column_perturbation, indicator=indicator_perturbation,
                                     column_cluster=column_cluster)
        for key, value in score.items():
            summary_performance[f'{fold}_key'] = value
        if 'top_100_genes' in score:
            rs_squared.append(score['top_100_genes']['r_squared'])
        else:
            rs_squared.append(score['all_genes']['r_squared'])
    summary_performance['avg_r_squared'] = sum(rs_squared) / len(rs_squared)
    return summary_performance