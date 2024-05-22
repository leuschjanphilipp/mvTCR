# A Variational Information Bottleneck Approach to Multi-Omics Data Integration
import torch
import torch.nn as nn
import scanpy as sc

from mvtcr.models.architectures.transformer import TransformerEncoder, TransformerDecoder
from mvtcr.models.architectures.mlp import MLP
from mvtcr.models.architectures.mlp_scRNA import build_mlp_encoder, build_mlp_decoder
from mvtcr.models.vae_base_model import VAEBaseModel
from mvtcr.dataloader.DataLoader import initialize_prediction_loader


class MoEModelTorch(nn.Module):
    def __init__(self, tcr_params, rna_params, vdj_params, joint_params):
        super(MoEModelTorch, self).__init__()
        #params list here
        xdim_rna = rna_params['xdim_rna']
        xdim_vdj = vdj_params['xdim_vdj']

        if alpha:
            self.alpha_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
            self.alpha_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

        if beta:
            self.beta_encoder = TransformerEncoder(tcr_params, hdim // self.amount_chains, num_seq_labels)
            self.beta_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

        if rna:
            self.rna_encoder = build_mlp_encoder(rna_params, xdim_rna, hdim)
            self.rna_decoder = build_mlp_decoder(rna_params, xdim_rna, hdim)

        if embeddings:
            if v_beta:
                self.embedding_v_beta = nn.Embedding(num_embeddings=input_dim['v_beta'], embedding_dim=embedding_dim)
            if d_beta:
                self.embedding_d_beta = nn.Embedding(num_embeddings=input_dim['d_beta'], embedding_dim=embedding_dim)
            if j_beta:
                self.embedding_j_beta = nn.Embedding(num_embeddings=input_dim['j_beta'], embedding_dim=embedding_dim)
            if v_alpha:
                self.embedding_v_alpha = nn.Embedding(num_embeddings=input_dim['v_alpha'], embedding_dim=embedding_dim)
            if j_alpha:
                self.embedding_j_alpha = nn.Embedding(num_embeddings=input_dim['j_alpha'], embedding_dim=embedding_dim)

        #can use mlp_scrna or should create mlp_vdj 
        self.vdj_encoder = build_mlp_encoder(vdj_params, xdim_vdj, hdim)
        self.vdj_decoder = build_mlp_decoder(vdj_params, xdim_vdj, hdim)


        self.tcr_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
                                   batch_norm, regularize_last_layer=False)
        self.tcr_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
                                   batch_norm, regularize_last_layer=True)

        self.rna_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
                                   batch_norm, regularize_last_layer=False)
        self.rna_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
                                   batch_norm, regularize_last_layer=True)
        
        self.vdj_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
                                   batch_norm, regularize_last_layer=False)
        self.vdj_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
                                   batch_norm, regularize_last_layer=True)

    
    def forward(self, rna, tcr, tcr_len, vdj, conditional=None):

        # Encode TCR 
        #TODO(if TCR)
        if beta:
            beta_seq = tcr[:, tcr.shape[1] // 2 * (self.amount_chains == 2):]
            beta_len = tcr_len[:, self.amount_chains - 1]
            h_beta = self.beta_encoder(beta_seq, beta_len) # shape=[batch_size, hdim//2]
        else:
            h_beta = placeholder

        if alpha:
            alpha_seq = tcr[:, :tcr.shape[1] // 2]
            alpha_len = tcr_len[:, 0]
            h_alpha = self.alpha_encoder(alpha_seq, alpha_len) # shape=[batch_size, hdim//2]
        else:
            h_beta = placeholder

        h_tcr = torch.cat([h_alpha, h_beta], dim=-1) # shape=[batch_size, hdim]

        # Encode RNA 
        #TODO(if RNA)
        h_rna = self.rna_encoder(rna)  # shape=[batch_size, hdim]

        # Encode VDJ 
        #TODO (if VDJ) shape and dim of vdj?
        h_vdj = self.vdj_encoder(vdj) # shape=[batch_size, hdim]


        # Predict Latent space
        z_rna_ = self.rna_vae_encoder(h_rna)  # shape=[batch_size, zdim*2]
        mu_rna, logvar_rna = z_rna_[:, :z_rna_.shape[1] // 2], z_rna_[:, z_rna_.shape[1] // 2:]
        z_rna = self.reparameterize(mu_rna, logvar_rna)  # shape=[batch_size, zdim]

        z_tcr_ = self.tcr_vae_encoder(h_tcr)  # shape=[batch_size, zdim*2]
        mu_tcr, logvar_tcr = z_tcr_[:, :z_tcr_.shape[1] // 2], z_tcr_[:, z_tcr_.shape[1] // 2:]
        z_tcr = self.reparameterize(mu_tcr, logvar_tcr)  # shape=[batch_size, zdim]

        z_vdj_ = self.vdj_vae_encoder(h_vdj) # shape=[batch_size, zdim*2]
        mu_vdj, logvar_vdj = z_vdj_[:, :z_vdj_.shape[1] // 2], z_vdj_[:, z_vdj_.shape[1] // 2:]
        z_vdj = self.reparametrize(mu_vdj, logvar_vdj)

        z = [z_rna, z_tcr, z_vdj]
        mu = [mu_rna, mu_tcr, mu_vdj]
        logvar = [logvar_rna, logvar_tcr, logvar_vdj]

        #Reconstruction
        rna_pred = []
        tcr_pred = []
        vdj_pred = []
        for z_ in z:
            # TODO if rna possible
            f_rna = self.rna_vae_decoder(z_)
            rna_pred.append(self.rna_decoder(f_rna))

            # TODO if tcr would do mandatory
            f_tcr = self.tcr_vae_decoder(z_)
            if beta:
                beta_pred = self.beta_decoder(f_tcr, beta_seq)
            else:
                beta_seq = torch.IntTensor()
            if alpha:
                alpha_pred = self.alpha_decoder(f_tcr, alpha_seq)
            else:
                alpha_pred =  torch.IntTensor()
            tcr_pred.append(torch.cat([alpha_pred, beta_pred], dim=1))


            # TODO if vdj
            f_vdj = self.vdj_vae_decoder(z_)
            vdj_pred.append(self.vdj_decoder(f_vdj))

        return z, mu, logvar, rna_pred, tcr_pred, vdj_pred
    
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

    def get_latent_from_z(self, z):
        return sum(z) / len(z)
    
class MoEModel(VAEBaseModel):
    def __init__(self,
                 adata,
                 params_architecture, 
                 balanced_sampling='clonotype',
                 metadata=None,
                 conditional=None,
                 optimization_mode_params=None,
                 label_key=None,
                 device=None
                 ):
        super(MoEModel, self).__init__(adata, params_architecture, balanced_sampling, metadata,
                                       conditional, optimization_mode_params, label_key, device)
        self.model_type = 'moe'

        self.params_tcr['max_tcr_length'] = adata.obsm['alpha_seq'].shape[1]
        self.params_tcr['num_seq_labels'] = len(self.aa_to_id)

        self.params_rna['xdim_rna'] = adata[0].X.shape[1]
        self.params_vdj['xdim_vdj'] = None # TODO params_vdj

        self.model = MoEModelTorch(self.params_tcr, self.params_rna, self.params_vdj, self.params_joint)

        def calculate_loss(self):
            pass

        def calculate_kld_loss(self,):
            pass

        def get_latent_unimodal(self,):
            pass
    
        def get_modality_contribution(self,):
            pass