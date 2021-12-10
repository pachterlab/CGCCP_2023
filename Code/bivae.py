# -*- coding: utf-8 -*-
"""Main module."""

from typing import Dict, Tuple
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi._compat import Literal

# from scvi.core.distributions import (
#     NegativeBinomial,
#     ZeroInflatedNegativeBinomial,
# )
# from ._base import DecoderSCVI, Encoder, LinearDecoderSCVI
# from .utils import one_hot

from scvi.core.modules._base import DecoderSCVI, Encoder
from scvi.core.modules.utils import one_hot

####
from scvi.core.modules import VAE
from distribution import BivariateNegativeBinomial

torch.backends.cudnn.benchmark = True

# VAE model
class BIVAE(VAE):
    """
    """
    def __init__(self,
                 gene_likelihood: str = "nb",
                 **kwargs):

        super().__init__(gene_likelihood=gene_likelihood,
                         **kwargs)

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

    def get_reconstruction_loss(
        self, x, px_rate, px_r, px_dropout, **kwargs
    ) -> torch.Tensor:
        # Reconstruction Loss
        if self.gene_likelihood == "nb":
            #### switched to BivariateNegativeBinomial
            reconst_loss = (
                -BivariateNegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        else:
            raise ValueError("Input valid gene_likelihood ['nb']")
        return reconst_loss
