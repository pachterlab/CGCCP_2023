# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Iterable, Optional

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


from scvi.module._vae import VAE

# import custom distributions 
from distributions import BivariateNegativeBinomial, log_prob_poisson, log_prob_NBcorr
from nnNB_module import log_prob_nnNB

torch.backends.cudnn.benchmark = True

# BIVAE model
class BIVAE(VAE):
    """
    """
    def __init__(self,
                 gene_likelihood: str = "nb",
                 mode: str = ['NB','NBcorr','Poisson','Bursty','custom'],
                 n_batch: int = 0,
                 n_continuous_cov: int = 0,
                 n_cats_per_cov: Optional[Iterable[int]] = None,
                 deeply_inject_covariates: bool = True,
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_size_factor_key: bool = False,
                 custom_dist = None,
                 THETA_IS : str ='NAS_SHAPE',
                 **kwargs):

        super().__init__(gene_likelihood=gene_likelihood,
                         n_continuous_cov=n_continuous_cov,
                         n_cats_per_cov=n_cats_per_cov,
                         **kwargs)
        
        
        # define the new custom distribution
        if mode == 'custom':
            self.custom_dist = custom_dist
        elif mode == 'NB':
            self.custom_dist = None
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
                                           THETA_IS = self.THETA_IS
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

        px_r = torch.exp(px_r)

#         if self.gene_likelihood == "zinb":
#             px = ZeroInflatedNegativeBinomial(
#                 mu=px_rate,
#                 theta=px_r,
#                 zi_logits=px_dropout,
#                 scale=px_scale,
#             )
        if self.gene_likelihood == "nb":
            px = BivariateNegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale, custom_dist = self.custom_dist,)
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