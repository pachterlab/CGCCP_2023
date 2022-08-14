# -*- coding: utf-8 -*-
"""Main module."""

from typing import Dict, Tuple
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi._compat import Literal


from scvi.core.modules._base import DecoderSCVI, Encoder
from scvi.core.modules.utils import one_hot

####
from scvi.core.modules import VAE
from distribution_2 import BivariateNegativeBinomial

torch.backends.cudnn.benchmark = True

# VAE model
class BIVAE(VAE):
    """
    """
    def __init__(self,
                 gene_likelihood: str = "nb",
                 mode: str = 'corr',
                 n_batch: int = 0,
                 n_continuous_cov: int = 0,
                 n_cats_per_cov: Optional[Iterable[int]] = None,
                 deeply_inject_covariates: bool = True,
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
                 use_size_factor_key: bool = False,
                 T: float = 1,
                 Trate: float = 1e-2,
                 Tmin: float = 1e-10,
                 custom_dist = None,
                 **kwargs):

        super().__init__(gene_likelihood=gene_likelihood,
                         # n_continuous_cov=n_continuous_cov,
                         # n_cats_per_cov=n_cats_per_cov,
                         **kwargs)

        self.T = T
        self.Trate = Trate
        self.Tmin = Tmin
        self.custom_dist = custom_dist

        if mode == 'mixed':
            self.mixed=True
            self.corr=True
        else:
            self.mixed=False
            self.corr=True
            if mode == 'uncorr':
                self.corr=False

            # elif mode == 'uncorr':
            #     self.corr=False
            # elif self.mode == "custom":
            #     self.corr = True

        #### switch to n_input/2 (shared between each spliced/unspliced gene)
        n_input = kwargs['n_input']
        n_input_px_r = int(n_input/2)

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

        # Modify decoder for mixed distribution prediction
        # decoder goes from n_latent-dimensional space to n_input-d data
        # For mixed, modify n_input to be n_input * 2 for
        n_latent = kwargs['n_latent']
        n_layers = kwargs['n_layers']
        n_hidden = kwargs['n_hidden']

        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_decoder = n_latent + n_continuous_cov
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        if self.mixed:
            # Multiply by 2 so half can be used for model weight
            n_output = n_input * 2
        else:
            n_output = n_input

        #### Fix: modify decoderSCVI class instead
        self.decoder = DecoderSCVI(
            n_latent,
            n_output,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )


    def get_reconstruction_loss(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        # Reconstruction Loss
        if self.gene_likelihood == "nb":
            #### switched to BivariateNegativ eBinomial
            reconst_loss = (
                -BivariateNegativeBinomial(mu=px_rate,
                                           theta=px_r,
                                           use_corr=self.corr,
                                           use_mixed=self.mixed,
                                           custom_dist=self.custom_dist,
                                           T=self.T,
                                           **kwargs).log_prob(x).sum(dim=-1)
            )
            # Update temperature
            if self.T > self.Tmin:
                self.T /= (1 + self.Trate)
            else:
                self.T = self.Tmin
        else:
            raise ValueError("Input valid gene_likelihood ['nb']")
        return reconst_loss