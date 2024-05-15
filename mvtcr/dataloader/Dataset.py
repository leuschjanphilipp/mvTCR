import torch
import numpy as np
from scipy import sparse


class JointDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			tcr_data,
			tcr_length,
			rna_data=None,
			vdj_data=None,
			citeseq_data=None,
			obs_metadata=None,
			labels=None,
			conditional=None
	):
		"""
		:param rna_data: list of gene expressions, where each element is a numpy or sparse matrix of one dataset
		:param tcr_data: list of seq_data, where each element is a seq_list of one dataset
		:param tcr_length: list of non-padded sequence length, needed for many architectures to mask the padding out
		:param vdj_data: list of vdj gene expression ohe encoded
		:param metadata: list of metadata
		:param labels: list of labels
		:param conditional: list of conditionales
		"""
		
		self.tcr_length = torch.LongTensor(tcr_length)
		self.tcr_data = torch.LongTensor(tcr_data)

		if rna_data is not None:
			self.rna_data = self._create_tensor(rna_data)
		else:
			self.rna_data = torch.BoolTensor([False] * self.tcr_data.shape[0])

		if vdj_data is not None:
			self.vjd_data = torch.LongTensor(vdj_data)
		else:
			self.vjd_data = torch.BoolTensor([False] * self.tcr_data.shape[0])
		
		if citeseq_data is not None:
			self.citeseq_data = torch.LongTensor(citeseq_data)
		else:
			self.citeseq_data = torch.BoolTensor([False] * self.tcr_data.shape[0])

		if obs_metadata is not None:
			self.metadata = obs_metadata.tolist()
		else:
			self.metadata = torch.BoolTensor([False] * self.tcr_data.shape[0])

		if labels is not None:
			self.labels = torch.LongTensor(labels)
		else:
			self.labels = torch.BoolTensor([False] * self.tcr_data.shape[0])

		if conditional is not None:
			# Reduce the one-hot-encoding back to labels
			self.conditional = torch.LongTensor(conditional.argmax(1))
		# LongTensor since it is going to be embedded
		else:
			self.conditional = torch.BoolTensor([False] * self.tcr_data.shape[0])


	def _create_tensor(self, x):
		if sparse.issparse(x):
			x = x.todense()
			return torch.FloatTensor(x)
		else:
			return torch.FloatTensor(x)

	def __len__(self):
		return self.tcr_data.shape[0]

	def __getitem__(self, idx):
		return self.tcr_data[idx], self.rna_data[idx], self.vjd_data[idx], self.citeseq_data[idx], \
			   self.labels[idx], self.conditional[idx]


class DeepTCRDataset(torch.utils.data.Dataset):
	def __init__(
			self,
			alpha_seq,
			beta_seq,
			vdj_dict,
			metadata
	):
		"""
		:param alpha_seq:
		:param beta_seq:
		:param vdj_dict:
		:param metadata: list of metadata
		"""
		self.metadata = np.concatenate(metadata, axis=0).tolist()

		# Concatenate datasets to be able to shuffle data through

		self.alpha_seq = np.concatenate(alpha_seq)
		self.alpha_seq = torch.LongTensor(self.alpha_seq)

		self.beta_seq = np.concatenate(beta_seq)
		self.beta_seq = torch.LongTensor(self.beta_seq)

		v_alpha = torch.LongTensor(vdj_dict['v_alpha'])
		j_alpha = torch.LongTensor(vdj_dict['j_alpha'])

		v_beta = torch.LongTensor(vdj_dict['v_beta'])
		d_beta = torch.LongTensor(vdj_dict['d_beta'])
		j_beta = torch.LongTensor(vdj_dict['j_beta'])

		self.vdj = torch.stack([v_alpha, j_alpha, v_beta, d_beta, j_beta], dim=1)

	def __len__(self):
		return len(self.alpha_seq)

	def __getitem__(self, idx):
		return self.alpha_seq[idx], self.beta_seq[idx], self.vdj[idx], self.metadata[idx]
