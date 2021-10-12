# A Variational Information Bottleneck Approach to Multi-Omics Data Integration
import torch
import torch.nn as nn
import numpy as np

from tcr_embedding.models.architectures.cnn import CNNEncoder, CNNDecoder
from tcr_embedding.models.architectures.bigru import BiGRUEncoder, BiGRUDecoder
from tcr_embedding.models.architectures.transformer import TransformerEncoder, TransformerDecoder
from tcr_embedding.models.architectures.mlp import MLP
from .mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from .vae_base_model import VAEBaseModel
from tcr_embedding.datasets.scdataset import TCRDataset


class PoEModelTorch(nn.Module):
	def __init__(self, xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm, seq_model_arch,
				 seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams, num_conditional_labels, cond_dim, cond_input=False):
		super(PoEModelTorch, self).__init__()

		seq_models = {'CNN': [CNNEncoder, CNNDecoder],
					  'Transformer': [TransformerEncoder, TransformerDecoder],
					  'BiGRU': [BiGRUEncoder, BiGRUDecoder]}

		scRNA_models = {'MLP': [build_mlp_encoder, build_mlp_decoder]}

		self.alpha_encoder = seq_models[seq_model_arch][0](seq_model_hyperparams, hdim//2, num_seq_labels)  # h//2 to avoid combined tcr dominate scRNA
		self.alpha_decoder = seq_models[seq_model_arch][1](seq_model_hyperparams, hdim, num_seq_labels)

		self.beta_encoder = seq_models[seq_model_arch][0](seq_model_hyperparams, hdim//2, num_seq_labels)
		self.beta_decoder = seq_models[seq_model_arch][1](seq_model_hyperparams, hdim, num_seq_labels)

		self.rna_encoder = scRNA_models[scRNA_model_arch][0](scRNA_model_hyperparams, xdim, hdim)
		self.rna_decoder = scRNA_models[scRNA_model_arch][1](scRNA_model_hyperparams, xdim, hdim)

		if cond_dim > 0:
			self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
		self.cond_input = cond_input
		cond_input_dim = cond_dim if cond_input else 0

		self.tcr_vae_encoder = MLP(hdim+cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.tcr_vae_decoder = MLP(zdim+cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		self.rna_vae_encoder = MLP(hdim+cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout, batch_norm, regularize_last_layer=False)  # zdim*2 because we predict mu and sigma simultaneously
		self.rna_vae_decoder = MLP(zdim+cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout, batch_norm, regularize_last_layer=True)

		# used for NB loss
		self.theta = torch.nn.Parameter(torch.randn(xdim))

	def forward(self, rna, tcr, tcr_len, conditional=None):
		"""
		Forward pass of autoencoder
		:param rna: torch.Tensor shape=[batch_size, num_genes]
		:param tcr: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:param tcr_len: torch.LongTensor shape=[batch_size] indicating how long the real unpadded length is
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
		:return:
			z: list of sampled latent variable zs. z = [z_rna, z_tcr, z_joint]
			mu: list of predicted means mu. mu = [mu_rna, mu_tcr, mu_joint]
			logvar: list of predicted logvars. logvar = [logvar_rna, logvar_tcr, logvar_joint]
			rna_pred: list of reconstructed rna. rna_pred = [rna_pred using z_rna, rna_pred using z_joint]
			tcr_pred: list of reconstructed tcr. tcr_pred = [tcr_pred using z_tcr, tcr_pred using z_joint]
		"""
		if conditional is not None:
			cond_emb_vec = self.cond_emb(conditional)
		# Encode TCR
		alpha_seq = tcr[:, :tcr.shape[1] // 2]
		alpha_len = tcr_len[:, 0]

		beta_seq = tcr[:, tcr.shape[1] // 2:]
		beta_len = tcr_len[:, 1]

		h_alpha = self.alpha_encoder(alpha_seq, alpha_len)  # shape=[batch_size, hdim//2]
		h_beta = self.beta_encoder(beta_seq, beta_len)  # shape=[batch_size, hdim//2]
		h_tcr = torch.cat([h_alpha, h_beta], dim=-1)  # shape=[batch_size, hdim]
		if conditional is not None and self.cond_input:
			h_tcr = torch.cat([h_tcr, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		# Encode RNA
		h_rna = self.rna_encoder(rna)  # shape=[batch_size, hdim]
		if conditional is not None and self.cond_input:
			h_rna = torch.cat([h_rna, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

		# Predict Latent space
		z_rna_ = self.rna_vae_encoder(h_rna)  # shape=[batch_size, zdim*2]
		mu_rna, logvar_rna = z_rna_[:, :z_rna_.shape[1]//2], z_rna_[:, z_rna_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z_rna = self.reparameterize(mu_rna, logvar_rna)  # shape=[batch_size, zdim]


		z_tcr_ = self.tcr_vae_encoder(h_tcr)  # shape=[batch_size, zdim*2]
		mu_tcr, logvar_tcr = z_tcr_[:, :z_tcr_.shape[1]//2], z_tcr_[:, z_tcr_.shape[1]//2:]  # mu.shape = logvar.shape = [batch_size, zdim]
		z_tcr = self.reparameterize(mu_tcr, logvar_tcr)  # shape=[batch_size, zdim]

		# Predict joint latent space using PoE
		mu_joint, logvar_joint = self.product_of_experts(mu_rna, mu_tcr, logvar_rna, logvar_tcr)
		z_joint = self.reparameterize(mu_joint, logvar_joint)

		z = [z_rna, z_tcr, z_joint]
		mu = [mu_rna, mu_tcr, mu_joint]
		logvar = [logvar_rna, logvar_tcr, logvar_joint]

		# Reconstruction
		rna_pred = []
		for z_ in [z_rna, z_joint]:
			if conditional is not None:
				z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
			f_rna = self.rna_vae_decoder(z_)
			rna_pred.append(self.rna_decoder(f_rna))
		tcr_pred = []
		for z_ in [z_tcr, z_joint]:
			if conditional is not None:
				z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
			f_tcr = self.tcr_vae_decoder(z_)
			alpha_pred = self.alpha_decoder(f_tcr, alpha_seq)
			beta_pred = self.beta_decoder(f_tcr, beta_seq)

			tcr_pred.append(torch.cat([alpha_pred, beta_pred], dim=1))

		return z, mu, logvar, rna_pred, tcr_pred

	def reparameterize(self, mu, log_var):
		"""
		https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
		std = torch.exp(0.5 * log_var)  # standard deviation
		eps = torch.randn_like(std)  # `randn_like` as we need the same size
		z = mu + (eps * std)  # sampling as if coming from the input space
		return z

	def product_of_experts(self, mu_rna, mu_tcr, logvar_rna, logvar_tcr):
		# formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
		logvar_joint = 1.0 / torch.exp(logvar_rna) + 1.0 / torch.exp(logvar_tcr) + 1.0  # sum up all inverse vars, logvars first needs to be converted to var, last 1.0 is coming from the prior
		logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar

		# formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, where mu_prior = 0.0
		mu_joint = mu_rna * (1.0 / torch.exp(logvar_rna)) + mu_tcr * (1.0 / torch.exp(logvar_tcr))
		mu_joint = mu_joint * torch.exp(logvar_joint)

		return mu_joint, logvar_joint

	def predict_transcriptome(self, z_shared, conditional=None):
		"""
		Predict the transcriptome connected to an shared latent space
		:param z_shared: torch.tensor, shared latent representation
		:param conditional:
		:return: torch.tensor, transcriptome profile
		"""
		if conditional is not None:  # more efficient than doing two concatenations
			cond_emb_vec = self.cond_emb(conditional)
			z_shared = torch.cat([z_shared, cond_emb_vec], dim=-1)  # shape=[batch_size, zdim+cond_dim]
		transcriptome_pred = self.rna_vae_decoder(z_shared)
		transcriptome_pred = self.rna_decoder(transcriptome_pred)
		return transcriptome_pred


class PoEModel(VAEBaseModel):
	def __init__(self,
				 adatas,  # adatas containing gene expression and TCR-seq
				 aa_to_id,
				 seq_model_arch,  # seq model architecture
				 seq_model_hyperparams,  # dict of seq model hyperparameters
				 scRNA_model_arch,
				 scRNA_model_hyperparams,
				 zdim,  # zdim
				 hdim,  # hidden dimension of encoder for each modality
				 activation,
				 dropout,
				 batch_norm,
				 shared_hidden=[],
				 names=[],
				 gene_layers=[],
				 seq_keys=[],
				 params_additional=None,
				 conditional=None,
				 optimization_mode='Reconstruction',
				 optimization_mode_params=None
				 ):
		super(PoEModel, self).__init__(adatas, aa_to_id, seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
									   zdim, hdim, activation, dropout, batch_norm, shared_hidden, names, gene_layers, seq_keys, params_additional, conditional,
									   optimization_mode=optimization_mode, optimization_mode_params=optimization_mode_params)

		self.poe = True
		seq_model_hyperparams['max_tcr_length'] = adatas[0].obsm['alpha_seq'].shape[1]
		xdim = adatas[0].X.shape[1] if self.gene_layers[0] is None else len(adatas[0].layers[self.gene_layers[0]].shape[1])
		num_seq_labels = len(aa_to_id)

		if self.conditional is not None:
			if self.conditional in adatas[0].obsm:
				num_conditional_labels = adatas[0].obsm[self.conditional].shape[1]
			else:
				num_conditional_labels = len(adatas[0].obs[self.conditional].unique())
			try:
				cond_dim = params_additional['c_embedding_dim']
			except:
				cond_dim = 20
		else:
			num_conditional_labels = 0
			cond_dim = 0

		self.model = PoEModelTorch(xdim, hdim, zdim, num_seq_labels, shared_hidden, activation, dropout, batch_norm,
								   seq_model_arch, seq_model_hyperparams, scRNA_model_arch, scRNA_model_hyperparams,
								   num_conditional_labels, cond_dim)

	def calculate_loss(self, scRNA_pred, scRNA, tcr_seq_pred, tcr_seq, loss_weights, scRNA_criterion, TCR_criterion, size_factor):
		''' Evaluate same-modality and joint reconstruction '''
		scRNA_loss = 0.5 * loss_weights[0] * \
					 (self.calc_scRNA_rec_loss(scRNA_pred[0], scRNA, scRNA_criterion, size_factor, self.losses[0]) +
					  self.calc_scRNA_rec_loss(scRNA_pred[1], scRNA, scRNA_criterion, size_factor, self.losses[0]))
		if tcr_seq_pred[0].shape[1] == tcr_seq.shape[1] - 2:  # For GRU and Transformer, as they don't predict start token for alpha and beta chain, so -2
			mask = torch.ones_like(tcr_seq).bool()
			mask[:, [0, mask.shape[1]//2]] = False
			TCR_loss = 0.5 * loss_weights[1] * \
					   (TCR_criterion(tcr_seq_pred[0].flatten(end_dim=1), tcr_seq[mask].flatten()) +
						TCR_criterion(tcr_seq_pred[1].flatten(end_dim=1), tcr_seq[mask].flatten()))
		else:  # For CNN, as it predicts start token
			TCR_loss = 0.5 * loss_weights[1] * \
					   (TCR_criterion(tcr_seq_pred[0].flatten(end_dim=1), tcr_seq.flatten()) +
						TCR_criterion(tcr_seq_pred[1].flatten(end_dim=1), tcr_seq.flatten()))

		loss = scRNA_loss + TCR_loss

		return loss, scRNA_loss, TCR_loss

	def create_datasets(self, adatas, names, layers, seq_keys, val_split, metadata=[], train_masks=None, label_key=None):
		"""
		Create torch Dataset, see above for the input
		:param adatas: list of adatas
		:param names: list of str
		:param layers:
		:param seq_keys:
		:param val_split:
		:param metadata:
		:param train_masks: None or list of train_masks: if None new train_masks are created, else the train_masks are used, useful for continuing training
		:return: train_dataset, val_dataset, train_masks (for continuing training)
		"""
		dataset_names_train = []
		dataset_names_val = []
		scRNA_datas_train = []
		scRNA_datas_val = []
		seq_datas_train = []
		seq_datas_val = []
		seq_len_train = []
		seq_len_val = []
		adatas_train = {}
		adatas_val = {}
		index_train = []
		index_val = []
		metadata_train = []
		metadata_val = []
		conditional_train = []
		conditional_val = []

		if train_masks is None:
			masks_exist = False
			train_masks = {}
		else:
			masks_exist = True

		# Iterates through datasets with corresponding dataset name, scRNA layer and TCR column key
		# Splits everything into train and val
		for i, (name, adata, layer, seq_key) in enumerate(zip(names, adatas, layers, seq_keys)):
			if masks_exist:
				train_mask = train_masks[name]
			else:
				if type(val_split) == str:
					train_mask = (adata.obs[val_split] == 'train').values
				else:
					# Create train mask for each dataset separately
					num_samples = adata.X.shape[0] if layer is None else len(adata.layers[layer].shape[0])
					train_mask = np.zeros(num_samples, dtype=np.bool)
					train_size = int(num_samples * (1 - val_split))
					if val_split != 0:
						train_mask[:train_size] = 1
					else:
						train_mask[:] = 1
					np.random.shuffle(train_mask)
				train_masks[name] = train_mask

			# Save dataset splits
			scRNA_datas_train.append(adata.X[train_mask] if layer is None else adata.layers[layer][train_mask])
			scRNA_datas_val.append(adata.X[~train_mask] if layer is None else adata.layers[layer][~train_mask])

			tcr_seq = np.concatenate([adata.obsm['alpha_seq'], adata.obsm['beta_seq']], axis=1)
			seq_datas_train.append(tcr_seq[train_mask])
			seq_datas_val.append(tcr_seq[~train_mask])

			seq_len = np.vstack([adata.obs['alpha_len'], adata.obs['beta_len']]).T

			seq_len_train += seq_len[train_mask].tolist()
			seq_len_val += seq_len[~train_mask].tolist()

			adatas_train[name] = adata[train_mask]
			adatas_val[name] = adata[~train_mask]

			dataset_names_train += [name] * adata[train_mask].shape[0]
			dataset_names_val += [name] * adata[~train_mask].shape[0]

			index_train += adata[train_mask].obs.index.to_list()
			index_val += adata[~train_mask].obs.index.to_list()

			metadata_train.append(adata.obs[metadata][train_mask].values)
			metadata_val.append(adata.obs[metadata][~train_mask].values)

			if self.conditional is not None:
				conditional_train.append(adata.obsm[self.conditional][train_mask])
				conditional_val.append(adata.obsm[self.conditional][~train_mask])
			else:
				conditional_train = None
				conditional_val = None

		train_dataset = TCRDataset(scRNA_datas_train, seq_datas_train, seq_len_train, adatas_train, dataset_names_train,
								   index_train, metadata_train, labels=None, conditional=conditional_train)
		if val_split != 0:
			val_dataset = TCRDataset(scRNA_datas_val, seq_datas_val, seq_len_val, adatas_val, dataset_names_val, index_val,
									 metadata_val, labels=None, conditional=conditional_val)
		else:
			val_dataset = None

		return train_dataset, val_dataset, train_masks