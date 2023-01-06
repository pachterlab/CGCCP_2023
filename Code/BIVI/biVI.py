import logging

from typing import Iterable, List, Optional, Dict, Sequence

from anndata import AnnData
import torch

from scvi._compat import Literal
# from scvi.core.data_loaders import ScviDataLoader
# from scvi.core.models import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin
# from scvi.core.modules import VAE
# from scvi.core.trainers import UnsupervisedTrainer
from scvi.model._scvi import SCVI

#### import the BIVAE model!
from .bivae import BIVAE

logger = logging.getLogger(__name__)

class biVI(SCVI):
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        **model_kwargs,
    ):
        ## switch from VAE to BIVAE
        super(SCVI, self).__init__(adata)
        self.module = BIVAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            **model_kwargs,
        )

        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())
