import logging

from typing import Iterable, List, Optional, Dict, Sequence

from anndata import AnnData
import torch
import numpy as np

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
            print(generate_outputs.keys())
            px = generative_outputs["px"]

            px_r = px.theta
            px_rate = px.mu
            assert px_rate.size[1] == 2000, f"px_rate size of 2000 expected, got: {px.size[1]}"
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
        print(self.mode)

        if self.module.gene_likelihood == "zinb":
            return_dict["dropout"] = dropout
            return_dict["dispersions"] = dispersions
        if self.module.gene_likelihood == "nb":
            return_dict["dispersions"] = dispersions
            print('gene likelihood nb, getting params')

        if self.module.mode == 'Bursty':
            print('Bursty mode, getting params')
            mu1 = means[:,:np.shape(params['mean'])[1]/2]
            mu2 = means[:,np.shape(params['mean'])[1]/2:]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions
            
            b,beta,gamma = get_bursty_params(mu1,mu2,dispersions)
            
            return_dict['burst_size'] = b
            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma
            
        if self.module.mode == 'NBcorr':
            mu1 = means[:,:np.shape(params['mean'])[1]/2]
            mu2 = means[:,np.shape(params['mean'])[1]/2:]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions
            
            alpha,beta,gamma = get_extrinsic_params(mu1,mu2,dispersions)
            
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
    
    def test_function(word_to_print):
        print(word_to_print)
        

def get_bursty_params(mu1,mu2,theta):
    ''' Returns b, beta, gamma of bursty distribution given mu1, mu2 and theta.
    Returns whatever size was input. 
    '''
    
    b = mu1/theta
    beta = 1/theta
    gamma = mu1/(mu2*theta)
    
    
    return(b,beta,gamma)


def get_extrinsic_params(mu1,mu2,theta):
    ''' Returns splicing rate beta, degradation rate gamma, and alpha (mean of transcription rate distribution) 
    given BVNB extrinsic noise model.
    '''
    alpha = theta
    beta = theta/mu1
    gamma = theta/mu2
    
    
    return(alpha,beta,gamma)

def get_constitutive_params(mu1,mu2):
    ''' Returns rate of splicing rate beta and rate of degradation gamma given constitutive model.
    '''
    beta = 1/mu1
    gamma = 1/mu2
    
    return(beta,gamma)
