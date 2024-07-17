import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import scanpy as sc
import anndata as ad
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
import operator

from .losses.kld import KLD
from mvtcr.dataloader.DataLoader import initialize_data_loader, initialize_latent_loader
from mvtcr.dataloader.DataLoader import initialize_prediction_loader
from mvtcr.utils_preprocessing import check_if_input_is_mudata

from .optimization.knn_prediction import report_knn_prediction
from .optimization.modulation_prediction import report_modulation_prediction
from .optimization.pseudo_metric import report_pseudo_metric


class VAEBaseModel(ABC):
	def __init__(self,
				 adata,
				 params_experiment,
				 params_architecture):
		"""
		VAE Base Model, used for both single and joint models
		:param adata: list of adatas containing train and val set
		:param conditional: str or None, if None a normal VAE is used, if str then the str determines the adata.obsm[conditional] as conditioning variable
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param balanced_sampling: None or str, indicate adata.obs column to balance
		:param optimization_mode_params: dict carrying the mode specific parameters
		"""
		self.adata = adata
		self.params_architecture = params_architecture
		self.balanced_sampling = params_experiment["balanced_sampling"]
		self.metadata = params_experiment["metadata"]
		self.conditional = params_experiment["conditional"]

		self.optimization_method = params_experiment["optimization_method"]
		self.prediction_key = params_experiment["prediction_key"]
		#self.optimization_mode_params = optimization_mode_params

		self.label_key = params_experiment["label_key"]
		self.device = params_experiment["device"]
		self.set_key = params_experiment["set_key"]

		self.params_tcr = params_architecture['tcr']
		self.params_rna = params_architecture['rna']
		self.params_vdj = None
		self.params_citeseq = None
		self.params_supervised = None
		self.params_joint = params_architecture['joint']
		#self.beta_only = False

		self.tcr_chain = params_experiment["tcr_chain"]
		self.use_vdj = False
		self.use_citeseq = False

		if params_experiment["use_vdj"]:
			self.params_vdj = params_architecture["vdj"]
			self.use_vdj = True
		if params_experiment["use_citeseq"]:
			self.params_citeseq = params_architecture["citeseq"]
			self.use_citeseq = True

		#TODO
		if 'supervised' in params_architecture:
			self.params_supervised = params_architecture['supervised']

		
		self.aa_to_id = adata.uns['aa_to_id']

		if self.device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self._train_history = defaultdict(list)
		self._val_history = defaultdict(list)

		# counters
		self.best_optimization_metric = None
		self.best_loss = None
		self.no_improvements = 0

		# loss functions
		self.loss_function_tcr = nn.CrossEntropyLoss(ignore_index=self.aa_to_id['_'])
		self.loss_function_rna = nn.MSELoss()
		self.loss_function_kld = KLD()
		self.loss_function_class = nn.CrossEntropyLoss()
		if self.use_vdj:
			self.loss_function_vdj = nn.CrossEntropyLoss()
		if self.use_citeseq:
			self.loss_function_citeseq = nn.MSELoss()

		# training params
		self.batch_size = params_architecture['batch_size']
		self.loss_weights = None
		self.comet = None
		self.kl_annealing_epochs = None

		# Model
		self.model_type = None
		self.model = None
		self.optimizer = None
		self.supervised_model = None
		if self.label_key is not None:
			self.supervised_model = self._build_supervised_head(label_key=self.label_key)

		# datasets
		if self.metadata is None:
			self.metadata = []
		if self.balanced_sampling is not None and self.balanced_sampling not in self.metadata:
			self.metadata.append(self.balanced_sampling)

		#adata, use_vdj, use_citeseq, metadata, conditional, label_key, balanced_sampling, batch_size
		self.data_train, self.data_val = initialize_data_loader(adata=adata,
														  		obs_set_key=self.set_key,
																tcr_chain=self.tcr_chain,
														  		use_vdj=self.use_vdj,
																use_citeseq=self.use_citeseq,
																metadata=self.metadata,
																conditional=self.conditional,
																label_key=self.label_key,
																balanced_sampling=self.balanced_sampling,
																batch_size=self.batch_size)


	def change_adata(self, new_adata):
		self.adata = new_adata
		self.aa_to_id = new_adata.uns['aa_to_id']
		if self.balanced_sampling is not None and self.balanced_sampling not in self.metadata:
			self.metadata.append(self.balanced_sampling)

		self.data_train, self.data_val = initialize_data_loader(new_adata, self.metadata, self.conditional,
																self.label_key,
																self.balanced_sampling, self.batch_size,
																beta_only=self.beta_only)

	def add_new_embeddings(self, num_new_embeddings):
		cond_emb_tmp = self.model.cond_emb.weight.data
		self.model.cond_emb = torch.nn.Embedding(cond_emb_tmp.shape[0] + num_new_embeddings, cond_emb_tmp.shape[1])
		self.model.cond_emb.weight.data[:cond_emb_tmp.shape[0]] = cond_emb_tmp

	def freeze_all_weights_except_cond_embeddings(self):
		"""
		Freezes conditional embedding weights to train in scArches style, since training data doesn't include
		previous labels, those embeddings won't be updated
		"""
		for param in self.model.parameters():
			param.requires_grad = False

		self.model.cond_emb.weight.requires_grad = True

	def unfreeze_all(self):
		for param in self.model.parameters():
			param.requires_grad = True

	def train(self,
			  n_epochs=200,
			  batch_size=512,
			  learning_rate=3e-4,
			  loss_weights=None,
			  kl_annealing_epochs=None,
			  early_stop=None,
			  save_path='../saved_models/',
			  comet=None):
		"""
		Train the model for n_epochs
		:param n_epochs: None or int, number of epochs to train, if None n_iters needs to be int
		:param batch_size: int, batch_size
		:param learning_rate: float, learning rate
		:param loss_weights: list of floats, loss_weights[0]:=weight or scRNA loss, loss_weights[1]:=weight for TCR loss, loss_weights[2] := KLD Loss
		:param kl_annealing_epochs: int or None, int number of epochs until kl reaches maximum warmup, if None this value is set to 30% of n_epochs
		:param early_stop: int, stop training after this number of epochs if val loss is not improving anymore
		:param save_path: str, path to directory to save model
		:param comet: None or comet_ml.Experiment object
		:return:
		"""
		self.batch_size = batch_size
		self.loss_weights = loss_weights
		self.comet = comet
		self.kl_annealing_epochs = kl_annealing_epochs
		#TODO assert 3 <= len(loss_weights) <= 4, 'Length of loss weights need to be either 3 or 4.'

		try:
			os.makedirs(save_path)  # Create directory to prevent Error while saving model weights
		except OSError:
			pass

		if kl_annealing_epochs is None:
			self.kl_annealing_epochs = int(0.3 * n_epochs)

		# raise ValueError(f'length of loss_weights must be 3, 4 (supervised) or None.')

		self.model = self.model.to(self.device)
		#if_supervised
		if self.supervised_model is not None:
			self.supervised_model = self.supervised_model.to(self.device)
		self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

		for epoch in tqdm(range(n_epochs)):
			self.model.train()
			#if_supervised
			if self.supervised_model is not None:
				self.supervised_model.train()
			train_loss_summary = self.run_epoch(epoch, phase='train')
			self.log_losses(train_loss_summary, epoch)

			self.model.eval()
			#if_supervised
			if self.supervised_model is not None:
				self.supervised_model.eval()
			with torch.no_grad():
				val_loss_summary = self.run_epoch(epoch, phase='val')
				self.log_losses(val_loss_summary, epoch)
				self.additional_evaluation(epoch, save_path)

			if self.do_early_stopping(val_loss_summary['val Loss'], early_stop, save_path, epoch):
				break

	def run_epoch(self, epoch, phase='train'):
		if phase == 'train':
			data = self.data_train
		else:
			data = self.data_val
		loss_total = []
		tcr_loss_total = []
		rna_loss_total = []
		vdj_loss_total = []
		citeseq_loss_total = []
		kld_loss_total = []
		cls_loss_total = []
		ys = []
		y_preds = []

		for batch in data:

			tcr, tcr_length, rna, vdj, citeseq, _, labels, conditional = batch.values()

			if rna.shape[0] == 1 and phase == 'train':
				continue  # BatchNorm cannot handle batches of size 1 during training phase

			tcr = tcr.to(self.device)
			tcr_length = tcr_length.to(self.device)
			rna = rna.to(self.device)
			vdj = vdj.to(self.device)
			citeseq = citeseq.to(self.device)
			labels = labels.to(self.device)
			conditional = conditional.to(self.device)

			z, mu, logvar, predictions = self.model.forward(tcr, tcr_length, rna, vdj, citeseq)
		
			kld_loss = self.calculate_kld_loss(mu, logvar, epoch)

			true = {"tcr": tcr, "rna": rna, "vdj": vdj, "citeseq": citeseq}
			loss_modalities = self.calculate_loss(predictions, true)
			
			loss = sum(loss_modalities.values()) + kld_loss

			if self.supervised_model is not None:
				#TODO
				y_pred = self.supervised_model(z)
				cls_loss = self.loss_function_class(y_pred, labels)
				loss += self.params_supervised['loss_weights_sv'] * cls_loss
				cls_loss_total.append(cls_loss)
				ys.append(labels.detach())
				y_preds.append(y_pred.detach())

			if phase == 'train':
				self.run_backward_pass(loss)

			loss_total.append(loss)
			rna_loss_total.append(loss_modalities["rna"])
			tcr_loss_total.append(loss_modalities["tcr"])
			if self.use_vdj:
				vdj_loss_total.append(loss_modalities["vdj"])
			if self.use_citeseq:
				citeseq_loss_total.append(loss_modalities["citeseq"])
			kld_loss_total.append(kld_loss)

			if torch.isnan(loss):
				print(f'ERROR: NaN in loss.')
				return

		loss_total = torch.stack(loss_total).mean().item()
		rna_loss_total = torch.stack(rna_loss_total).mean().item()
		tcr_loss_total = torch.stack(tcr_loss_total).mean().item()
		kld_loss_total = torch.stack(kld_loss_total).mean().item()
		if self.use_vdj:
			vdj_loss_total = torch.stack(vdj_loss_total).mean().item()
		if self.use_citeseq:
			citeseq_loss_total = torch.stack(citeseq_loss_total).mean().item()

		summary_losses = {f'{phase} Loss': loss_total,
						  f'{phase} TCR Loss': tcr_loss_total,
						  f'{phase} RNA Loss': rna_loss_total,
						  f'{phase} KLD Loss': kld_loss_total}
		if self.use_vdj:
			summary_losses[f'{phase} VDJ Loss'] = vdj_loss_total
		if self.use_citeseq:
			summary_losses[f'{phase} CiteSeq Loss'] = citeseq_loss_total

		#if_supervised
		if self.supervised_model is not None:
			#TODO
			cls_loss_total = torch.stack(cls_loss_total).mean().item()
			summary_losses[f'{phase} CLS Loss'] = cls_loss_total
			ys = torch.cat(ys, dim=0)
			y_preds = torch.cat(y_preds, dim=0)
			summary_losses[f'{phase} CLS F1'] = f1_score(y_preds.argmax(1).detach().cpu(), ys.cpu(), average='weighted')

		self.summary_losses = summary_losses

		return summary_losses

	def log_losses(self, summary_losses, epoch):
		if self.comet is not None:
			self.comet.log_metrics(summary_losses, epoch=epoch)

	def run_backward_pass(self, loss):
		self.optimizer.zero_grad()
		loss.backward()
		#TODO
		#if self.optimization_mode_params is not None and 'grad_clip' in self.optimization_mode_params:
		#	nn.utils.clip_grad_value_(self.model.parameters(), self.optimization_mode_params['grad_clip'])
		self.optimizer.step()

	def additional_evaluation(self, epoch, save_path):
		if self.optimization_method is None:
			return
		
		name = self.optimization_method
		if name == 'reconstruction':
			return
		if name == 'knn_prediction':
			score, relation = report_knn_prediction(self.adata, self, self.prediction_key,
													epoch, self.comet)
		elif name == 'modulation_prediction':
			#TODO check for ambiguity
			score, relation = report_modulation_prediction(self.adata, self, self.optimization_mode_params, #here
														   epoch, self.comet)
		elif name == 'pseudo_metric':
			score, relation = report_pseudo_metric(self.adata, self, self.prediction_key,
												   epoch, self.comet)
		elif name == 'supervised':
			#TODO
			score, relation = self.summary_losses['val CLS F1'], operator.gt
		else:
			raise ValueError('Unknown Optimization mode')
		#if self.best_optimization_metric is None or relation(score, self.best_optimization_metric):
		#	self.best_optimization_metric = score
		#	self.save(os.path.join(save_path, f'best_model_by_metric.pt'))
		#if self.comet is not None:
		#	self.comet.log_metric('max_metric', self.best_optimization_metric, epoch=epoch)

	def do_early_stopping(self, val_loss, early_stop, save_path, epoch):
		if self.best_loss is None or val_loss < self.best_loss:
			self.best_loss = val_loss
			self.save(os.path.join(save_path, 'best_model_by_reconstruction.pt'))
			self.no_improvements = 0
		else:
			self.no_improvements += 1
		if early_stop is not None and self.no_improvements > early_stop:
			print('Early stopped')
			return True
		if self.comet is not None:
			self.comet.log_metric('Epochs without Improvements', self.no_improvements, epoch=epoch)
		return False

	# <- prediction functions ->
	@check_if_input_is_mudata
	def get_latent(self, adata, metadata, return_mean=True, copy_adata_obs=False):
		#TODO write that adata ist first arg followed by kwargs because of mudata decorator
		"""
		Get latent
		:param adata:
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""		
		data_embed = initialize_prediction_loader(adata, 
												tcr_chain=self.tcr_chain,
												use_vdj=self.use_vdj, 
												use_citeseq=self.use_citeseq, 
												metadata=metadata,
												batch_size=self.batch_size, 
												conditional=self.conditional)
		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()

			for batch in data_embed:
				tcr, tcr_length, rna, vdj, citeseq, metadata, labels, conditional = batch.values()
								
				tcr = tcr.to(self.device)
				tcr_length = tcr_length.to(self.device)
				rna = rna.to(self.device)
				vdj = vdj.to(self.device)
				citeseq = citeseq.to(self.device)
				conditional = conditional.to(self.device)
				
				z, mu, logvar, predictions = self.model.forward(tcr, tcr_length, rna, vdj, citeseq)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None

				if return_mean:
					z = mu
				z = self.model.get_latent_from_z(z)
				z = sc.AnnData(z.detach().cpu().numpy())
				# z.obs[metadata] = np.array(metadata_batch).T
				zs.append(z)
		latent = sc.AnnData.concatenate(*zs)
		latent.obs.index = adata.obs.index
		
		for key in metadata:
			latent.obs[key] = adata.obs[key]
		if copy_adata_obs:
				latent.obs = adata.obs.copy()
		return latent
	
	#TODO here decorator as well?
	def get_all_latent(self, adata, metadata, return_mean=True):
		"""
		Get latent
		:param adata:
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
		data_embed = initialize_prediction_loader(adata, metadata, self.batch_size, beta_only=self.beta_only,
												  conditional=self.conditional)

		zs = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			for rna, tcr, seq_len, _, labels, conditional in data_embed:
				rna = rna.to(self.device)
				tcr = tcr.to(self.device)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None
				z, mu, _, _, _ = self.model(rna, tcr, seq_len, conditional)
				if return_mean:
					z = mu
				zs.append(z)
		return zs

	def predict_rna_from_latent(self, adata_latent, metadata=None):
		data = initialize_latent_loader(adata_latent, self.batch_size, self.conditional)
		rnas = []
		with torch.no_grad():
			model = self.model.to(self.device)
			model.eval()
			for batch in data:
				if self.conditional is not None:
					batch = batch[0].to(self.device)
					conditional = batch[1].to(self.device)
				else:
					batch = batch[0].to(self.device)
					conditional = None
				batch_rna = model.predict_transcriptome(batch, conditional)
				batch_rna = sc.AnnData(batch_rna.detach().cpu().numpy())
				rnas.append(batch_rna)
		rnas = sc.AnnData.concatenate(*rnas)
		rnas.obs.index = adata_latent.obs.index
		if metadata is not None:
			rnas.obs[metadata] = adata_latent.obs[metadata]
		return rnas

	def predict_label(self, adata, use_mean=True):
		data_embed = initialize_prediction_loader(adata, [], self.batch_size, beta_only=self.beta_only,
												  conditional=self.conditional)

		ys = []
		with torch.no_grad():
			self.model = self.model.to(self.device)
			self.model.eval()
			self.supervised_model = self.supervised_model.to(self.device)
			self.supervised_model.eval()
			for rna, tcr, seq_len, _, labels, conditional in data_embed:
				rna = rna.to(self.device)
				tcr = tcr.to(self.device)
				seq_len = seq_len.to(self.device)

				if self.conditional is not None:
					conditional = conditional.to(self.device)
				else:
					conditional = None
				z, mu, _, _, _ = self.model(rna, tcr, seq_len, conditional)
				if use_mean:
					z = mu
				z = self.model.get_latent_from_z(z)
				y = self.supervised_model(z)
				ys.append(y)
		ys = torch.cat(ys, dim=0)

		return ys

	# <- semi-supervised model ->
	def _build_supervised_head(self, label_key):
		assert self.params_supervised is not None, 'Please specify parameters for supervised model'

		hidden_neurons = [self.params_supervised['hidden_neurons']] * self.params_supervised['num_hidden_layers']
		hidden_neurons = [self.params_joint['zdim']] + hidden_neurons + [
			max(7, self.adata.obs[label_key].unique().max() + 1)]

		layers = []
		for i in range(len(hidden_neurons) - 2):
			layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]))
			if self.params_supervised['batch_norm']:
				layers.append(nn.BatchNorm1d(hidden_neurons[i + 1]))
			if self.params_supervised['activation'] == 'relu':
				layers.append(nn.ReLU())
			elif self.params_supervised['activation'] == 'tanh':
				layers.append(nn.Tanh())
			elif self.params_supervised['activation'] == 'sigmoid':
				layers.append(nn.Sigmoid())
			elif self.params_supervised['activation'] == 'linear':
				pass
			else:
				raise ValueError(f'Invalid activation: {self.params_supervised["activation"]}')
			if self.params_supervised['dropout'] > 0:
				layers.append(nn.Dropout(self.params_supervised['dropout']))

		layers.append(nn.Linear(hidden_neurons[-2], hidden_neurons[-1]))
		return nn.Sequential(*layers)

	def forward_supervised(self, z):
		z_ = self.model.get_latent_from_z(z)
		prediction = self.supervised_model(z_)
		return prediction

	# <- loss functions ->
	@abstractmethod
	def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
		raise NotImplementedError

	@abstractmethod
	def calculate_kld_loss(self, mu, logvar, epoch):
		"""
		Calculate the kld loss and z depending on the model type
		:param mu: mean of the VAE latent space
		:param logvar: log(variance) of the VAE latent space
		:param epoch: current epoch as integer
		:return:
		"""
		#TODO check for ambiguity
		raise NotImplementedError('Implement this in the different model versions')

	def calculate_classification_loss(self, prediction, labels):
		loss = self.loss_function_class(prediction, labels)
		loss = self.params_supervised['loss_weights_sv'] * loss
		return loss

	def get_kl_annealing_factor(self, epoch):
		"""
		Calculate KLD annealing factor, i.e. KLD needs to get warmup
		:param epoch: current epoch
		:return:
		"""
		return min(1.0, epoch / self.kl_annealing_epochs)

	# <- logging helpers ->
	@property
	def history(self):
		return pd.DataFrame(self._val_history)

	@property
	def train_history(self):
		return pd.DataFrame(self._train_history)

	def save(self, filepath):
		""" Save model and optimizer state, and auxiliary data for continuing training """
		model_file = {'state_dict': self.model.state_dict(),
					  'train_history': self._train_history,
					  'val_history': self._val_history,
					  'aa_to_id': self.aa_to_id,

					  'params_architecture': self.params_architecture,
					  'balanced_sampling': self.balanced_sampling,
					  'metadata': self.metadata,
					  'conditional': self.conditional,
					  'optimization_mode_params': self.optimization_mode_params, #TODO
					  'label_key': self.label_key,
					  'model_type': self.model_type,
					  }
		if self.supervised_model is not None:
			model_file['supervised_model'] = self.supervised_model.state_dict()
		torch.save(model_file, filepath)

	def load(self, filepath, map_location=torch.device('cuda')):
		""" Load model for evaluation / inference"""
		model_file = torch.load(os.path.join(filepath), map_location=map_location)
		self.model.load_state_dict(model_file['state_dict'], strict=False)
		if 'supervised_model' in model_file:
			self.supervised_model.load_state_dict(model_file['supervised_model'], strict=True)
		self._train_history = model_file['train_history']
		self._val_history = model_file['val_history']
		self.aa_to_id = model_file['aa_to_id']
