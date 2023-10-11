import sys
sys.path.append('../')

import logging
from typing import Iterable, List, Optional, Dict, Sequence, Union, TypeVar, Tuple
from collections.abc import Iterable as IterableClass
Number = TypeVar("Number", int, float)

from anndata import AnnData
import torch
import numpy as np
import pandas as pd

from scvi._compat import Literal
# from scvi.core.data_loaders import ScviDataLoader
# from scvi.core.models import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin
# from scvi.core.modules import VAE
# from scvi.core.trainers import UnsupervisedTrainer
from scvi.model._scvi import SCVI
from scvi.model._utils import (
    _get_batch_code_from_category,
)

#### import the BIVAE model!
from BIVI import bivae

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
        decoder_type : Literal["non-linear","linear"] = "non-linear",
        **model_kwargs,
    ):
        ## switch from VAE to BIVAE
        super(SCVI, self).__init__(adata)
        self.module = bivae.BIVAE(
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
            decoder_type = decoder_type,
            **model_kwargs,
        )

        self._model_summary_string = (
            "BIVI Model with the following params: \n mode: {}, n_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            self.module.mode,
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
            px = generative_outputs["px"]

            px_r = px.theta
            px_rate = px.mu
    
            if self.module.gene_likelihood == "zinb":
                px_dropout = px.zi_probs

            n_batch = px_rate.size()[0] if n_samples == 1 else px_rate.size(1)

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
        n_genes = np.shape(means)[-1]/2
  


        if self.module.gene_likelihood == "zinb":
            return_dict["dropout"] = dropout
            return_dict["dispersions"] = dispersions
        if self.module.gene_likelihood == "nb":
            return_dict["dispersions"] = dispersions


        if self.module.mode == 'Bursty':
            print('Bursty mode, getting parameters')
            mu1 = means[...,:int(n_genes)]
            mu2 = means[...,int(n_genes):]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions

            
            b,beta,gamma = get_bursty_params(mu1,mu2,dispersions,THETA_IS = self.module.THETA_IS)
            
            return_dict['burst_size'] = b
            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma
            
        if self.module.mode == 'NBcorr':
            print('Extrinsic mode, getting parameters')
            
            mu1 = means[...,:int(n_genes)]
            mu2 = means[...,int(n_genes):]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            return_dict['dispersions'] = dispersions
            
            alpha,beta,gamma = get_extrinsic_params(mu1,mu2,dispersions)
            
            return_dict['alpha'] = alpha
            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma
            
        if self.module.mode == 'Poisson':
            print('Constitutive mode, getting parameters')
            mu1 = means[...,:int(n_genes)]
            mu2 = means[...,int(n_genes):]
            return_dict['unspliced_means'] = mu1
            return_dict['spliced_means'] = mu2
            
            beta,gamma = 1/mu1,1/mu2

            return_dict['rel_splicing_rate'] = beta
            return_dict['rel_degradation_rate'] = gamma

        return return_dict
    
    
    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata=None,
        indices=None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Optional[Union[float, Literal["latent"]]] = 1,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = False,
#         return_numpy: Optional[bool] = None,
    ) -> Dict[str, np.array]:
        r"""
        Returns the normalized gene expression, normalized burst size, and relative degradation rate.


        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean  -- VALID
            Whether to return the mean of the samples.


        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is numpy array.
        """
        adata = self._validate_anndata(adata)
        post = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            # FOR NOW!!! Change so that gene list and what genes in anndata are called are consistent
            all_genes = adata.var['gene_name'].tolist()
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if indices is None:
            indices = np.arange(adata.n_obs)

#         if n_samples > 1 and return_mean is False:
#             if return_numpy is False:
#                 logger.warning(
#                     "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
#                 )
#             return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch)
        
        scale_list_gene = []
        dispersion_list = []

        for tensors in post:
            x = tensors['X']
            px_scale = torch.zeros_like(x)
      
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])

            for b in transform_batch:
                if b is not None:
                    batch_indices = tensors[_CONSTANTS.BATCH_KEY]
                    tensors[_CONSTANTS.BATCH_KEY] = torch.ones_like(batch_indices) * b
                inference_kwargs = dict(n_samples=n_samples)
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    compute_loss=False,
                )
                px = generative_outputs["px"]
                
                if library_size == "latent":
                    px_scale += px.rate.cpu()
                else:
                    px_scale += px.scale.cpu()
                    
                px_scale = px.scale[..., gene_mask]
                
                

            px_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)
            
            px_theta = generative_outputs['px'].theta
            dispersion_list.append(px_theta)

            
        return_dict = {}


        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            dispersion_list = torch.cat(dispersion_list,dim=1).cpu().numpy()
            n_genes = int(scale_list_gene.size()[-1]/2)
            scale_list_unspliced = scale_list_gene[...,:n_genes].cpu().numpy()
            scale_list_spliced = scale_list_gene[...,n_genes:].cpu().numpy()
            
            if self.module.mode == "Bursty":
                b,beta,gamma = get_bursty_params(scale_list_unspliced,scale_list_spliced,
                                                 dispersion_list, THETA_IS = self.module.THETA_IS)
                return_dict['norm_burst_size'] = b
                return_dict['norm_degradation_rate'] = gamma
                return_dict['norm_splicing_rate'] = beta
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            elif self.module.mode == "NBcorr":
                alpha,beta,gamma = get_extrinsic_params(scale_list_unspliced,scale_list_spliced,dispersion_list)
                return_dict['norm_alpha'] = alpha
                return_dict['norm_beta'] = beta
                return_dict['norm_gamma'] = gamma
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            elif self.module.mode == "Poisson":
                beta,gamma = get_constitutive_params(scale_list_unspliced,scale_list_spliced)
                return_dict['norm_splicing_rate'] = beta
                return_dict['norm_degradation_rate'] = gamma
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            else:
                raise Exception("Please use valid biVI mode: Bursty, NBcorr, or Poisson.")
        
        
        else:
            scale_list_genes = torch.cat(scale_list_gene, dim=0)
            dispersion_list = torch.cat(dispersion_list,dim=0).cpu().numpy()
            n_genes = int(np.shape(scale_list_genes)[-1]/2)
            scale_list_unspliced = scale_list_genes[...,:n_genes].cpu().numpy()
            scale_list_spliced = scale_list_genes[...,n_genes:].cpu().numpy()
            
            if self.module.mode == "Bursty":
                b,beta,gamma = get_bursty_params(scale_list_unspliced,scale_list_spliced,
                                                 dispersion_list,THETA_IS = self.module.THETA_IS)
                return_dict['norm_burst_size'] = b
                return_dict['norm_degradation_rate'] = gamma
                return_dict['norm_splicing_rate'] = beta
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            elif self.module.mode == "NBcorr":
                alpha,beta,gamma = get_extrinsic_params(scale_list_unspliced,scale_list_spliced,dispersion_list)
                return_dict['norm_alpha'] = alpha
                return_dict['norm_beta'] = beta
                return_dict['norm_gamma'] = gamma
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            elif self.module.mode == "Poisson":
                beta,gamma = get_constitutive_params(scale_list_unspliced,scale_list_spliced)
                return_dict['norm_splicing_rate'] = beta
                return_dict['norm_degradation_rate'] = gamma
                return_dict['norm_spliced_mean'] = scale_list_spliced
                return_dict['norm_unspliced_mean'] = scale_list_unspliced
            else:
                raise Exception("Please use valid biVI mode: Bursty, NBcorr, or Poisson.")

        

        if (return_mean == True) and (n_samples > 1):
            for param in return_dict.keys():
                return_dict[param] = np.mean(return_dict[param], axis=0)
        
        return return_dict
    
    def get_bayes_factors(
        self,
        adata = None,
        idx1 = None,
        idx2 = None,
        delta : Optional[float] = 0.2,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Optional[Union[float, Literal["latent"]]] = 1,
        n_samples_1 : int = 10,
        n_samples_2 : int = 10,
        n_comparisons: int = 5000,
        batch_size : Optional[int] = None,
        return_all_lfc : bool = False,
        # potentially change
        eps : float = 1e-10,
        return_df : bool = False,
        params_dict_1 : str = 'Calculate',
        params_dict_2 : str = 'Calculate'
    ) -> Union[pd.DataFrame,Tuple[Dict[str, np.array],Dict[str,np.array]]]:
    
        ''' Calculates Bayes Factor for gene_list (or all genes if no gene_list is input).
        Considers two hypotheses for differential expression of parameters in groups A and B:
        
        LFC : $lfc = log2(\rho_a) - log2(\rho_b)$
        
        $H_0 = P(|lfc| >= delta)$
        $H_1 = P(|lfc| < delta)$
        
        Parameters
        -------------------
        idx1 
            index for group A
        idx2
            index for group B
        delta
            threshold above which to consider the LFC differential between two parameters
        gene_list
            genes to consider for differential expression testing
        library size
            default 1 "normalized", can scale or use "latent"
        n_samples_1
            number of samples from posterior to take for each cell in sample 1
        n_samples_2
            number of samples from posterior to take for each cell in sample 2
        n_comparisons
            number of permuted comparisons between samples in each cell,max is (|idx1|*n_samples_1*|idx2|*n_samples_2)             
        batch_size
            size of batch to pass through forward model when sampling from posterior
        return_all_lfc
            return all the calculated LFCs, defaults to returning the median
        return_df
            return pandas DataFrame with information
            
    
        Returns
        -------------------
        BF_dict
            dictionary of all calculated Bayes Factors for genes between groups A and B
        effect_size_dict
            dictionary with either mean or array of all effect sizes for genes between A and B
        
        or
        bayes_df
            pandas DataFrame with information
        '''
        
        BF_dict = {}
        effect_size_dict = {}
        df_dict = {}
        
        if params_dict_1 == 'Calculate':
        # sample from posterior for ind1 and ind2 -- doesn't deal with batches yet, could add that later
            params_dict_1 = self.get_normalized_expression(
            adata=adata,
            indices = idx1,
    #         transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list = gene_list,
            library_size = library_size,
            n_samples = n_samples_1,
            batch_size = batch_size,
            return_mean = False,
            )
        
        if params_dict_2 == 'Calculate':
            params_dict_2 = self.get_normalized_expression(
            adata=adata,
            indices = idx2,
    #         transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list = gene_list,
            library_size = library_size,
            n_samples = n_samples_2,
            batch_size = batch_size,
            return_mean = False,
            )
        
    
        n_possible_permutations = len(idx1)*len(idx2)*n_samples_1*n_samples_2
        n_comparisons = min(n_comparisons,n_possible_permutations)
    
        # Bayes Factor for each parameter
    
        for param in params_dict_1.keys():
            
            params1 = params_dict_1[param]
            params2 = params_dict_2[param]
            
  
            N_genes = params1.shape[-1]
            
            # reshape
            params1 = params1.reshape(len(idx1)*n_samples_1,N_genes)
            params2 = params2.reshape(len(idx2)*n_samples_2,N_genes)
        
        
            compare_array1,compare_array2 = get_compare_arrays(params1,params2,n_comparisons)
        
            # a la scVI -- shrink LFC to 0 when there are no observed nascent or maturecounts
            where_zero_a = (np.max(adata[idx1].X,0).todense() == 0)[:,:N_genes] & (np.max(adata[idx1].X,0).todense() == 0)[:,N_genes:]
            where_zero_b = (np.max(adata[idx2].X,0).todense() == 0)[:,:N_genes] & (np.max(adata[idx2].X,0).todense() == 0)[:,N_genes:]

        
            eps = self.estimate_pseudocounts_offset(params1, params2, where_zero_a, where_zero_b)
        
        
            lfc_values = np.log2(compare_array1+eps) - np.log2(compare_array2+eps)
            lfc_abs = np.abs(lfc_values)
            
 
        
            BF = np.sum(lfc_abs>=delta,axis=0)/(np.sum(lfc_abs<delta,axis=0))
            BF_dict[param] = BF
 
        
            if return_all_lfc == True:
                effect_size_dict[param] = lfc_values
            else:
                effect_size_dict[param] = np.median(lfc_values,axis=0)
            
            if return_df == True:
                param_dict = {}
                param_dict['prob_DE'] = np.sum(lfc_abs>=delta,axis=0)
                param_dict['prob_not_DE'] = np.sum(lfc_abs<delta,axis=0)
                param_dict['bayes_factor'] = BF
                param_dict['lfc_mean'] = np.mean(lfc_values,axis=0)
                param_dict['lfc_median'] = np.median(lfc_values,axis=0)
                param_dict['lfc_std'] = np.std(lfc_values,axis=0)
                param_dict['lfc_min'] = np.min(lfc_values,axis=0)
                param_dict['lfc_max'] = np.max(lfc_values,axis=0)
                param_dict['delta'] = [delta]*np.shape(lfc_values)[-1]
                df_dict[param] = pd.DataFrame(param_dict)
                
        
        if return_df == False:
            return BF_dict, effect_size_dict
        else:
            return df_dict
        
    def estimate_pseudocounts_offset(self,
                                     scales_a,
                                     scales_b,
                                     where_zero_a,
                                     where_zero_b,
                                     percentile= 0.9
                                                    ):
        """
        **ADAPTED FROM SCVI**
        Determines pseudocount offset.
        This shrinks LFCs asssociated with non-expressed genes to zero.
        Parameters
        ----------
        scales_a
            Scales in first population
        scales_b
            Scales in second population
        where_zero_a
            mask where no observed counts
        where_zero_b
            mask where no observed counts
        """
        max_scales_a = np.max(scales_a, 0).reshape(1,-1)
        max_scales_b = np.max(scales_b, 0).reshape(1,-1)
        asserts = (
            (max_scales_a.shape == where_zero_a.shape)
            and (max_scales_b.shape == where_zero_b.shape)
        ) and (where_zero_a.shape == where_zero_b.shape)
        if not asserts:
            raise ValueError(
            "Dimension mismatch between scales and/or masks to compute the pseudocounts offset."
            )
        if where_zero_a.sum() >= 1:
            artefact_scales_a = max_scales_a[where_zero_a]
            eps_a = np.percentile(artefact_scales_a, q=percentile)
        else:
            eps_a = 1e-10

        if where_zero_b.sum() >= 1:
            artefact_scales_b = max_scales_b[where_zero_b]
            eps_b = np.percentile(artefact_scales_b, q=percentile)
        else:
            eps_b = 1e-10
        res = np.maximum(eps_a, eps_b)
        return res



def get_compare_arrays(params1,params2,n_comparisons):
    ''' Returns comparison arrays for params1 and params2.
    
    Randomly samples from params1 and params2 n_comparison times and constructs two arrays of the random samples. 
    '''
    length_1 = np.shape(params1)[0]
    length_2 = np.shape(params2)[0]
    n_each_sample = min(length_1,length_2)
    n_left_to_sample = n_comparisons

    compare_array1 = np.zeros((n_comparisons,params1.shape[1]))
    compare_array2 = np.zeros((n_comparisons,params1.shape[1]))

    # how many have been sampled
    n_sampled = 0
    
    while n_left_to_sample > 0:
    
        if n_left_to_sample > n_each_sample:
            samp = n_each_sample
        if n_left_to_sample < n_each_sample:
            samp = n_left_to_sample
        
    
        rand_1 = np.random.choice(length_1,samp)
        rand_2 = np.random.choice(length_2,samp)
    
        arr1 = params1[rand_1,:]
        arr2 = params2[rand_2,:]
        
   
        
        compare_array1[n_sampled : n_sampled+samp, :] = arr1
        compare_array2[n_sampled : n_sampled+samp, :]  = arr2
    
        n_left_to_sample -= samp
        n_sampled += samp
        
    return compare_array1,compare_array2

def get_bursty_params(mu1,mu2,theta,THETA_IS = 'NAS_SHAPE'):
    ''' Returns b, beta, gamma of bursty distribution given mu1, mu2 and theta.
    Returns whatever size was input. 
    '''
    
    if THETA_IS == 'MAT_SHAPE':
        gamma = 1/theta
        b = mu2*gamma
        beta = b/mu1

    elif THETA_IS == 'B':
        b = theta
        beta = b/mu1
        gamma = b/mu2
        
    elif THETA_IS == 'NAS_SHAPE':
        beta = 1/theta
        b = mu1*beta
        gamma = b/mu2

    
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

