# -*- coding: utf-8 -*-
"""Main module."""
"""Built atop scVI-tools https://github.com/scverse/scvi-tools/tree/7523a30c16397620cf50098fb0fa53cd32395090"""
import sys
sys.path.append('../')

from typing import Dict, Iterable, Optional, Sequence, Union
import anndata
from anndata import AnnData

import numpy as np
import torch
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi._types import LatentDataType
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseLatentModeModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, LinearDecoderSCVI, one_hot

torch.backends.cudnn.benchmark = True

from scvi.module._vae import VAE, LDVAE

# import custom distributions
from BIVI.distributions import BivariateNegativeBinomial, log_prob_poisson, log_prob_NBcorr, log_prob_NBuncorr
from BIVI.nnNB_module import log_prob_nnNB

torch.backends.cudnn.benchmark = True

# BIVAE model
class BIVAE(VAE):
    """
    """
    def __init__(self,
                 gene_likelihood: str = "nb",
                 mode: Literal['NB','NBcorr','Poisson','Bursty','custom'] = 'Bursty',
                 n_batch: int = 0,
                 n_continuous_cov: int = 0,
                 n_cats_per_cov: Optional[Iterable[int]] = None,
                 deeply_inject_covariates: bool = True,
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_size_factor_key: bool = False,
                 custom_dist = None,
                 THETA_IS :  Literal['NAS_SHAPE','MAT_SHAPE','B'] ='NAS_SHAPE',
                 decoder_type : Literal["non-linear","linear"] = "non-linear",
                 bias : bool = False,
                 **kwargs):
        print(kwargs)
        super().__init__(gene_likelihood=gene_likelihood,
                         n_continuous_cov=n_continuous_cov,
                         n_cats_per_cov=n_cats_per_cov,
                         **kwargs)

        self.decoder_type = decoder_type
        self.mode = mode
        print('Initiating biVAE')
        print(f'Mode: {mode}, Decoder: {decoder_type}, Theta is: {THETA_IS}')
        
        # define the new custom distribution
        if mode == 'custom':
            self.custom_dist = custom_dist
        elif mode == 'NB':
            self.custom_dist = log_prob_NBuncorr
        elif mode == 'NBcorr':
            self.custom_dist = log_prob_NBcorr
        elif mode == 'Poisson':
            self.custom_dist = log_prob_poisson
        elif mode == 'Bursty':
            self.custom_dist = log_prob_nnNB
        self.THETA_IS = THETA_IS

        #### switch to n_input/2 (shared between each spliced/unspliced gene)
        n_input = kwargs['n_input']
        n_input_px_r = int(n_input/2) # theta !!

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_px_r))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_px_r, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_px_r, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )


        # decoder goes from n_latent-dimensional space to n_input-d data
        n_latent = kwargs['n_latent']
        n_layers = kwargs['n_layers']
        n_hidden = kwargs['n_hidden']

        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_decoder = n_latent + n_continuous_cov
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        n_output = n_input

        #### modify decoderSCVI class
        if decoder_type == "non-linear":
            self.decoder = DecoderSCVI(
                n_input_decoder,
                n_output, # modified
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
                )
        elif decoder_type == "linear":
            self.decoder = LinearDecoderSCVI(
            n_latent,
            n_input,
            n_cat_list=cat_list,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            bias=bias,
            )
        
      

    # redefine the reconstruction error
    def get_reconstruction_loss(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        # Reconstruction Loss
        if self.gene_likelihood == "nb":
            #### switch to BivariateNegative Binomial
            reconst_loss = (
                -BivariateNegativeBinomial(mu=px_rate,
                                           theta=px_r,
                                           custom_dist=self.custom_dist,
                                           THETA_IS = self.THETA_IS,
                                           dispersion = self.dispersion,
                                           mode = self.mode,
                                           **kwargs).log_prob(x).sum(dim=-1)
            )

        else:
            raise ValueError("Input valid gene_likelihood ['nb']")
        return reconst_loss

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # CHANGED FOR BIVAE
        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )


        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
#         elif self.dispersion == "gene-cell":
#             px_r = self.px_r

        px_r = torch.exp(px_r)

#         if self.gene_likelihood == "zinb":
#             px = ZeroInflatedNegativeBinomial(
#                 mu=px_rate,
#                 theta=px_r,
#                 zi_logits=px_dropout,
#                 scale=px_scale,
#             )
        if self.gene_likelihood == "nb":
            px = BivariateNegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale, 
                                           custom_dist = self.custom_dist,
                                           THETA_IS = self.THETA_IS,
                                           dispersion = self.dispersion,
                                           mode = self.mode)
#         elif self.gene_likelihood == "NegativeBinomial":
#             px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
# #         elif self.gene_likelihood == "poisson":
#             px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(
            px=px,
            pl=pl,
            pz=pz,
        )

    @torch.inference_mode()
    def get_likelihood_parameters(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = 1,
        give_mean: Optional[bool] = False,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""
        Estimates for the parameters of the likelihood :math:`p(x \mid z)`
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        give_mean
            Return expected value of parameters or a samples
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        dropout_list = []
        mean_list = []
        dispersion_list = []
        
        
        for tensors in scdl:
            inference_kwargs = dict(n_samples=n_samples)
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

            px = generative_outputs["px"]

            px_r = px.theta
            px_rate = px.mu
 
            if self.module.gene_likelihood == "zinb":
                px_dropout = px.zi_probs

            n_batch = px_rate.size(0) if n_samples == 1 else px_rate.size(1)

            px_r = px_r.cpu().numpy()
            if len(px_r.shape) == 1:
                dispersion_list += [np.repeat(px_r[np.newaxis, :], n_batch, axis=0)]
            else:
                dispersion_list += [px_r]
            mean_list += [px_rate.cpu().numpy()]
            if self.module.gene_likelihood == "zinb":
                dropout_list += [px_dropout.cpu().numpy()]
                dropout = np.concatenate(dropout_list, axis=-2)
                
        means = np.concatenate(mean_list, axis=-2)
        dispersions = np.concatenate(dispersion_list, axis=-2)

        if give_mean and n_samples > 1:
            if self.module.gene_likelihood == "zinb":
                dropout = dropout.mean(0)
            means = means.mean(0)
            dispersions = dispersions.mean(0)

        return_dict = {}
        return_dict["mean"] = means


        if self.module.gene_likelihood == "zinb":
            return_dict["dropout"] = dropout
            return_dict["dispersions"] = dispersions
        if self.module.gene_likelihood == "nb":
            return_dict["dispersions"] = dispersions
            print('gene likelihood nb, getting params')

        if self.module.mode == 'Bursty':
            print('Bursty mode, returning parameters')
            mu1 = means[:,:int(np.shape(means)[1]/2)]
            mu2 = means[:,int(np.shape(means)[1]/2):]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions
            
            b,beta,gamma = get_bursty_params(mu1,mu2,dispersions,THETA_IS = self.module.THETA_IS)
            
            return_dict['burst_size'] = b
            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma
            
        if self.module.mode == 'NBcorr':
            mu1 = means[:,:np.shape(params['mean'])[1]/2]
            mu2 = means[:,np.shape(params['mean'])[1]/2:]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions
            
            alpha,beta,gamma = get_extrinsic_params(mu1,mu2,dispersions,THETA_IS = self.module.THETA_IS)
            
            return_dict['alpha'] = alpha
            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma
            
        if self.module.mode == 'Poisson':
            mu1 = means[:,:np.shape(params['mean'])[1]/2]
            mu2 = means[:,np.shape(params['mean'])[1]/2:]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            
            beta,gamma = 1/mu1,1/mu2

            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma

        return return_dict

        

