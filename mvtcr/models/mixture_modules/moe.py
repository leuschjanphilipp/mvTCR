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
    def __init__(self, tcr_params, rna_params, vdj_params, citeseq_params, joint_params):
        super(MoEModelTorch, self).__init__()
        
        self.both_tcr_chains = True if tcr_params['tcr_chain'] == "both" else False
        self.use_vdj = True if vdj_params is not None else False
        self.use_citeseq = True if citeseq_params is not None else False
        
        #self.amount_chains = 1 if self.tcr_chain != "both" else 2

        num_seq_labels = tcr_params['num_seq_labels']

        xdim = rna_params['xdim']

        if self.use_vdj:
            #TODO could do down to avoid double if, but cleaner here for now so params and modules separated
            emb_dim = vdj_params["vdj_embedding_dim"]
            num_v_alpha_labels = vdj_params["num_v_alpha_labels"]
            num_j_alpha_labels = vdj_params["num_j_alpha_labels"]
            num_v_beta_labels = vdj_params["num_v_beta_labels"]
            num_d_beta_labels = vdj_params["num_d_beta_labels"]
            num_j_beta_labels = vdj_params["num_j_beta_labels"]
        
        if self.use_citeseq:
            pass

        hdim = joint_params['hdim']
        num_conditional_labels = joint_params['num_conditional_labels']
        cond_dim = joint_params['cond_dim']
        cond_input = joint_params['cond_input']
        zdim = joint_params['zdim']
        shared_hidden = joint_params['shared_hidden']
        activation = joint_params['activation']
        dropout = joint_params['dropout']
        batch_norm = joint_params['batch_norm']
        use_embedding_for_cond = joint_params['use_embedding_for_cond'] if 'use_embedding_for_cond' in joint_params else True

        # used for NB loss
        self.theta = torch.nn.Parameter(torch.randn(xdim))

        #if both chains; second transformer
        if self.both_tcr_chains:
            self.alpha_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
            self.alpha_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

            self.beta_encoder = TransformerEncoder(tcr_params, hdim // 2, num_seq_labels)
            self.beta_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)
        else:
            self.beta_encoder = TransformerEncoder(tcr_params, hdim, num_seq_labels)
            self.beta_decoder = TransformerDecoder(tcr_params, hdim, num_seq_labels)

        self.rna_encoder = build_mlp_encoder(rna_params, xdim, hdim)
        self.rna_decoder = build_mlp_decoder(rna_params, xdim, hdim)

        #conditional
        if cond_dim > 0:
            if use_embedding_for_cond:
                self.cond_emb = torch.nn.Embedding(num_conditional_labels, cond_dim)
            else:  # use one hot encoding
                self.cond_emb = None
                cond_dim = num_conditional_labels
                
        self.cond_input = cond_input
        self.use_embedding_for_cond = use_embedding_for_cond
        self.num_conditional_labels = num_conditional_labels
        cond_input_dim = cond_dim if cond_input else 0

        self.tcr_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
                                   batch_norm, regularize_last_layer=False)
        self.tcr_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
                                   batch_norm, regularize_last_layer=True)

        self.rna_vae_encoder = MLP(hdim + cond_input_dim, zdim * 2, shared_hidden, activation, 'linear', dropout,
                                   batch_norm, regularize_last_layer=False)
        self.rna_vae_decoder = MLP(zdim + cond_dim, hdim, shared_hidden[::-1], activation, activation, dropout,
                                   batch_norm, regularize_last_layer=True)

        #VDJ modality
        if self.use_vdj:
            self.embedding_v_alpha = nn.Embedding(num_embeddings=num_v_alpha_labels, embedding_dim=emb_dim)
            self.embedding_j_alpha = nn.Embedding(num_embeddings=num_j_alpha_labels, embedding_dim=emb_dim)
            self.embedding_v_beta = nn.Embedding(num_embeddings=num_v_beta_labels, embedding_dim=emb_dim)
            self.embedding_d_beta = nn.Embedding(num_embeddings=num_d_beta_labels, embedding_dim=emb_dim)
            self.embedding_j_beta = nn.Embedding(num_embeddings=num_j_beta_labels, embedding_dim=emb_dim)

            self.vdj_encoder = build_mlp_encoder(vdj_params, emb_dim*5, hdim)
            self.vdj_decoder = build_mlp_decoder(vdj_params, emb_dim*5, hdim)
            
            self.vdj_vae_encoder = MLP(hdim + cond_input_dim, 
                                       zdim * 2, 
                                       shared_hidden, 
                                       activation, 
                                       'linear', 
                                       dropout,
                                       batch_norm, 
                                       regularize_last_layer=False)
            
            self.vdj_vae_decoder = MLP(zdim + cond_dim, 
                                       hdim, 
                                       shared_hidden[::-1], 
                                       activation, 
                                       activation, 
                                       dropout,
                                       batch_norm, 
                                       regularize_last_layer=True)
            #TODO softmax

        # CiteSeq modality
        if self.use_citeseq:
            pass


    def forward(self, tcr, tcr_len, rna, vdj, citeseq, conditional=None):
        """
		Forward pass of autoencoder
		:param rna: torch.Tensor shape=[batch_size, num_genes]
		:param tcr: torch.Tensor shape=[batch_size, seq_len, feature_dim]
		:param tcr_len: torch.Tensor shape=[batch_size]
		:param conditional: torch.Tensor shape=[batch_size, n_cond] one-hot-encoded conditional covariates
		:return:
			z: list of sampled latent variable zs. z = [z_rna, z_tcr, z_joint]
			mu: list of predicted means mu. mu = [mu_rna, mu_tcr, mu_joint]
			logvar: list of predicted logvars. logvar = [logvar_rna, logvar_tcr, logvar_joint]
			rna_pred: list of reconstructed rna. rna_pred = [rna_pred using z_rna, rna_pred using z_joint]
			tcr_pred: list of reconstructed tcr. tcr_pred = [tcr_pred using z_tcr, tcr_pred using z_joint]
		"""
        # Encode TCR
        if self.both_tcr_chains:
            #TODO check if tcr = cat(a, b) or cat(b, a)
            alpha_seq =  tcr[:, tcr.shape[1] // 2:]
            alpha_len = tcr_len[:, 0]
            beta_seq = tcr[:, :tcr.shape[1] // 2]
            beta_len = tcr_len[:, 1]

            h_beta = self.beta_encoder(beta_seq, beta_len)  # shape=[batch_size, hdim//2]
            h_alpha = self.alpha_encoder(alpha_seq, alpha_len)  # shape=[batch_size, hdim//2]
            h_tcr = torch.cat([h_alpha, h_beta], dim=-1)  # shape=[batch_size, hdim]
        else:
            #TODO check input to encoder
            h_beta = self.beta_encoder(tcr, tcr_len)  # shape=[batch_size, hdim//2]
            h_tcr = torch.cat([h_beta], dim=-1)  # shape=[batch_size, hdim]

        # Encode RNA
        h_rna = self.rna_encoder(rna)  # shape=[batch_size, hdim]

        if conditional is not None and self.cond_input:
            cond_emb_vec = self.cond_emb(conditional) if self.use_embedding_for_cond else \
                torch.nn.functional.one_hot(conditional, self.num_conditional_labels)

            h_tcr = torch.cat([h_tcr, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
            h_rna = torch.cat([h_rna, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

        # Predict Latent space
        z_tcr_ = self.tcr_vae_encoder(h_tcr)  # shape=[batch_size, zdim*2]
        mu_tcr, logvar_tcr = z_tcr_[:, :z_tcr_.shape[1] // 2], z_tcr_[:, z_tcr_.shape[1] // 2:]
        z_tcr = self.reparameterize(mu_tcr, logvar_tcr)  # shape=[batch_size, zdim]

        z_rna_ = self.rna_vae_encoder(h_rna)  # shape=[batch_size, zdim*2]
        mu_rna, logvar_rna = z_rna_[:, :z_rna_.shape[1] // 2], z_rna_[:, z_rna_.shape[1] // 2:]
        z_rna = self.reparameterize(mu_rna, logvar_rna)  # shape=[batch_size, zdim]

        z = {"rna": z_rna, "tcr": z_tcr}
        mu = {"rna": mu_rna, "tcr": mu_tcr}
        logvar = {"rna": z_rna, "tcr": z_tcr}

        if self.use_vdj:
            # Encode VDJ
            h_v_alpha = self.embedding_v_alpha(vdj[:,0])
            h_j_alpha = self.embedding_j_alpha(vdj[:,1])
            h_v_beta = self.embedding_v_beta(vdj[:,2])
            h_d_beta = self.embedding_d_beta(vdj[:,3])
            h_j_beta = self.embedding_j_beta(vdj[:,4])
            h_vdj = self.vdj_encoder(torch.cat((h_v_alpha, h_j_alpha, h_v_beta, h_d_beta, h_j_beta), dim=-1))
            # Conditional
            if conditional is not None and self.cond_input:
                h_vdj = torch.cat([h_vdj, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]
            # Predict Latent space
            z_vdj_ = self.vdj_vae_encoder(h_vdj) # shape=[batch_size, zdim*2]
            mu_vdj, logvar_vdj = z_vdj_[:, :z_vdj_.shape[1] // 2], z_vdj_[:, z_vdj_.shape[1] // 2:]
            z_vdj = self.reparameterize(mu_vdj, logvar_vdj)
            z["vdj"] = z_vdj
            mu["vdj"] = mu_vdj
            logvar["vdj"] = logvar_vdj
        
        #TODO
        if self.use_citeseq:
            # Encode CiteSeq
            pass
            # Conditional
            # Predict latent space

        #Reconstruction
        predictions = dict()
        for z_ in z.values():
            # Conditional
            if conditional is not None:
                z_ = torch.cat([z_, cond_emb_vec], dim=1)  # shape=[batch_size, hdim+n_cond]

            # TCR
            f_tcr = self.tcr_vae_decoder(z_)
            if self.both_tcr_chains:
                beta_pred = self.beta_decoder(f_tcr, beta_seq)
                alpha_pred = self.alpha_decoder(f_tcr, alpha_seq)
                tcr_pred = torch.cat([alpha_pred, beta_pred], dim=1)
            else:
                tcr_pred = self.beta_decoder(f_tcr, beta_seq)
            predictions["tcr"] = tcr_pred

    	    # RNA
            f_rna = self.rna_vae_decoder(z_)
            predictions["rna"] = self.rna_decoder(f_rna)

            # VDJ
            if self.use_vdj:
                f_vdj = self.vdj_vae_decoder(z_)
                predictions["vdj"] = self.vdj_decoder(f_vdj)

            # CiteSeq
            if self.use_citeseq:
                predictions["citeseq"] = None

        return z, mu, logvar, predictions


    def reparameterize(self, mu, logvar):
        """
		https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
        #mu = torch.stack(list(mu.values()))
        #logvar = torch.stack(list(logvar.values()))
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        z = mu + (std * eps)  # sampling as if coming from the input space
        return z
    
    def get_latent_from_z(self, z):
        # z.sum(axis=0) / n 
        return torch.stack(list(z.values())).sum(axis=0) / len(z)

    def predict_modality_from_shared_latent(self, z_shared, modality, conditional=None, tcr_seq=None):
        if conditional is not None:  # more efficient than doing two concatenations
            cond_emb_vec = self.cond_emb(conditional)
            z_shared = torch.cat([z_shared, cond_emb_vec], dim=-1)  # shape=[batch_size, zdim+cond_dim]

        if modality == "tcr":
            #TODO
            f_tcr = self.tcr_vae_decoder(z_shared)
            if self.amount_chains != 1:
                alpha_pred = self.alpha_decoder(f_tcr, tcr_seq[0])
                beta_pred = self.beta_decoder(f_tcr, tcr_seq[1])
                tcr_pred = torch.cat([alpha_pred, beta_pred], dim=1)
            else:
                tcr_pred = self.beta_decoder(f_tcr, tcr_seq)
            prediction = tcr_pred
        elif modality == "rna":
            f_rna = self.rna_vae_decoder(z_shared)
            prediction = self.rna_decoder(f_rna)
        elif modality == "vdj":
            f_vdj = self.vdj_vae_decoder(z_shared)
            prediction = self.vdj_decoder(f_vdj)
        elif modality == "citeseq":
            pass
        else:
            print("Modality not found. Chose one of {'tcr', 'rna', 'vdj', 'citeseq'}.")
            prediction = None
        return prediction


class MoEModel(VAEBaseModel):
    def __init__(self,
                 adata,
                 params_experiment,
                 params_architecture):
        super(MoEModel, self).__init__(adata, params_experiment, params_architecture)
        self.model_type = 'moe'

        self.params_tcr["tcr_chain"] = params_experiment["tcr_chain"]
        self.params_tcr['max_tcr_length'] = adata.obsm['alpha_seq'].shape[1]
        self.params_tcr['num_seq_labels'] = len(self.aa_to_id)

        self.params_rna['xdim'] = adata[0].X.shape[1]

        num_conditional_labels = 0
        cond_dim = 0
        if self.conditional is not None:
            if self.conditional in adata.obsm:
                num_conditional_labels = adata.obsm[self.conditional].shape[1]
            else:
                num_conditional_labels = len(adata.obs[self.conditional].unique())
            if 'c_embedding_dim' not in self.params_joint:
                cond_dim = 20
            else:
                cond_dim = self.params_joint['c_embedding_dim']
        self.params_joint['num_conditional_labels'] = num_conditional_labels
        self.params_joint['cond_dim'] = cond_dim
        self.params_joint['cond_input'] = self.conditional is not None

        if self.use_vdj: 
            #number of labels in categorical col
            self.params_vdj["num_v_alpha_labels"] = adata.obs["VJ_1_v_call"].max() + 1
            self.params_vdj["num_j_alpha_labels"] = adata.obs["VJ_1_j_call"].max() + 1
            self.params_vdj["num_v_beta_labels"] = adata.obs["VDJ_1_v_call"].max() + 1
            self.params_vdj["num_d_beta_labels"] = adata.obs["VDJ_1_d_call"].max() + 1
            self.params_vdj["num_j_beta_labels"] = adata.obs["VDJ_1_j_call"].max() + 1
        
        if self.use_citeseq:
            #TODO
            pass

        self.model = MoEModelTorch(self.params_tcr, self.params_rna, self.params_vdj, self.params_citeseq, self.params_joint)

    def calculate_loss(self, rna_pred, rna, tcr_pred, tcr):
        rna_loss = self.loss_function_rna(rna_pred[0], rna) + self.loss_function_rna(rna_pred[1], rna)
        rna_loss *= 0.5 * self.loss_weights[0]

        # For GRU and Transformer, as they don't predict start token for alpha and beta chain, so amount of chains used
        if tcr_pred[0].shape[1] == tcr.shape[1] - self.model.amount_chains:
            mask = torch.ones_like(tcr).bool()
            mask[:, [0]] = False
            if not self.beta_only:
                mask[:, [mask.shape[1] // 2]] = False
            tcr_loss = self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr[mask].flatten())
            if not 'beta_only' in self.params_tcr or self.params_tcr['beta_only']:
                tcr_loss += self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr[mask].flatten())
            tcr_loss *= 0.5 * self.loss_weights[1]
        else:  # For CNN, as it predicts start token
            tcr_loss = (self.loss_function_tcr(tcr_pred[0].flatten(end_dim=1), tcr.flatten()) +
                        self.loss_function_tcr(tcr_pred[1].flatten(end_dim=1), tcr.flatten()))
            tcr_loss *= 0.5 * self.loss_weights[1]
        return rna_loss, tcr_loss

    def calculate_kld_loss(self, mu, logvar, epoch):

        
        kld_loss = (self.loss_function_kld(mu[0], logvar[0]) + self.loss_function_kld(mu[1], logvar[1]))
        kld_loss *= 0.5 * self.loss_weights[2] * self.get_kl_annealing_factor(epoch)
        z = 0.5 * (mu[0] + mu[1])
        return kld_loss, z

    def get_latent_unimodal(self, adata, metadata, modality, return_mean=True):
        """
		Get unimodal latent either from RNA or TCR only
		:param adata:
		:param metadata: list of str, list of metadata that is needed, not really useful at the moment
		:param return_mean: bool, calculate latent space without sampling
		:return: adata containing embedding vector in adata.X for each cell and the specified metadata in adata.obs
		"""
        data_embed = initialize_prediction_loader(adata, metadata, self.batch_size, beta_only=self.beta_only)

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
                if modality == 'RNA':
                    z = z[0]
                else:
                    z = z[1]
                z = sc.AnnData(z.detach().cpu().numpy())
                # z.obs[metadata] = np.array(metadata_batch).T
                zs.append(z)
        latent = sc.AnnData.concatenate(*zs)
        latent.obs.index = adata.obs.index
        latent.obs[metadata] = adata.obs[metadata]
        return latent

    def get_modality_contribution(self, adata):
        import numpy as np

        def angular_similarity(x, y):
            dot = np.dot(x, y)
            norms = np.linalg.norm(x) * np.linalg.norm(y)
            cos_similarity = dot / norms
            angular = (1 - np.arccos(cos_similarity) / np.pi)
            return angular

        unimodal_latent = self.get_all_latent(adata, metadata=[], return_mean=True)
        rna_latent = [batch[0].detach().cpu().numpy() for batch in unimodal_latent]
        rna_latent = np.vstack(rna_latent)
        rna_latent = sc.AnnData(X=rna_latent)

        tcr_latent = [batch[1].detach().cpu().numpy() for batch in unimodal_latent]
        tcr_latent = np.vstack(tcr_latent)
        tcr_latent = sc.AnnData(X=tcr_latent)

        joint_latent = sc.AnnData(X=(tcr_latent.X + rna_latent.X) * 0.5)
        ang_tcr = np.array([angular_similarity(x, y) for x, y in zip(joint_latent.X, tcr_latent.X)])
        ang_rna = np.array([angular_similarity(x, y) for x, y in zip(joint_latent.X, rna_latent.X)])
        adata.obs['contribution_tcr-rna'] = ang_tcr - ang_rna + 0.5
