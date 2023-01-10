

# Math
import numpy as np
import pandas as pd

# scvi
import anndata
import scanpy as sc


# stats
from scipy.stats import ttest_ind
from scipy.stats import ranksums

def get_bursty_params(mu1,mu2,theta):
    ''' Returns b, beta, gamma of bursty distribution given mu1, mu2 and theta.
    Returns whatever size was input. 
    '''
    
    b = mu1/theta
    beta = 1/theta
    gamma = mu1/(mu2*theta)
    
    
    return(b,beta,gamma)


def perform_ttest(adata,genes,results_dict,cluster_label='subclass_label',setup = 'Bursty-10-NAS_SHAPE',
                   alternative = ['two-sided','less','greater'],
                   permutations = None,subset_index=False,
                   return_dataframe = False,
                   gene_names = 'gene_name'):
    ''' Perform three ttests on genes in adata using parameters in results dict. 
    '''
    
    # subset the results_dict and adata if subset != False
    if subset_index != False:
        adata = adata[subset_index]
        norm_means = np.array(results_dict['Bursty-10-NAS_SHAPE']['norm_means'])[subset_index]
        dispersions = results_dict[setup]['params']['dispersions'][subset_index]
    else:
        norm_means = np.array(results_dict['Bursty-10-NAS_SHAPE']['norm_means'])
        dispersions = results_dict[setup]['params']['dispersions']
        
    # define dictionaries to hold t and p values
    b_dict = {}
    gamma_dict = {}
    mu1_dict = {}
    mu2_dict = {}
    
    b_dict['genes'] = genes
    gamma_dict['genes'] = genes
    mu1_dict['genes'] = genes
    mu2_dict['genes'] = genes
    
    # define cluster labels
    cluster_labels = np.array(adata.obs[cluster_label].tolist())
    unique_clusters = np.unique(cluster_labels)
    adata_genes = adata.var[gene_names].tolist()
        
    N = int(len(results_dict[setup]['params']['mean'][0,:])/2)
    
    # arrays to hold parameters
    b = np.ones((len(adata),len(genes)))
    gamma = np.ones((len(adata),len(genes)))
    mu1 = np.ones((len(adata),len(genes)))
    mu2 = np.ones((len(adata),len(genes)))

    
    # loop through genes and calculate parameters
    for i,g in enumerate(genes):
        index = adata_genes.index(g)
        mu1_ = norm_means[:,index]
        mu2_ =  norm_means[:,index+N]
        theta = dispersions[:,index]
    
        b_,beta_,gamma_ = get_bursty_params(mu1_,mu2_,theta)
        b[:,i] = b_
        gamma[:,i] = gamma_
        
        mu1[:,i] = mu1_
        mu2[:,i] = mu2_
            
    for cl in unique_clusters:
        print(cl)
        # subset parameters
        b_cl = b[cluster_labels == cl,:]
        gamma_cl = gamma[cluster_labels == cl,:]
        mu1_cl = mu1[cluster_labels == cl,:]
        mu2_cl = mu2[cluster_labels == cl,:]
        
        b_rest = b[cluster_labels != cl,:]
        gamma_rest = gamma[cluster_labels != cl,:]
        mu1_rest = mu1[cluster_labels != cl,:]
        mu2_rest = mu2[cluster_labels != cl,:]
        
        
        # fold change: av (normalized param in cl/normalized param in rest)
        b_dict[f'{cl} fold change'] = np.mean(b_cl,axis=0)/np.mean(b_rest,axis=0)
        gamma_dict[f'{cl} fold change'] = np.mean(gamma_cl,axis=0)/np.mean(gamma_rest,axis=0)
        mu1_dict[f'{cl} fold change'] = np.mean(mu1_cl,axis=0)/np.mean(mu1_rest,axis=0)
        mu2_dict[f'{cl} fold change'] = np.mean(mu2_cl,axis=0)/np.mean(mu2_rest,axis=0)
        
        # statistics for each parameters
        for alt in alternative:
            statistics_b, pvalues_b = ttest_ind(b_cl, b_rest, axis=0, equal_var= False, 
                              alternative=alt,permutations=permutations)
            statistics_gamma, pvalues_gamma = ttest_ind(gamma_cl, gamma_rest, axis=0, equal_var= False, 
                              alternative=alt,permutations=permutations)
            statistics_mu1, pvalues_mu1 = ttest_ind(mu1_cl, mu1_rest, axis=0, equal_var= False, 
                              alternative=alt,permutations=permutations)
            statistics_mu2, pvalues_mu2 = ttest_ind(mu2_cl, mu2_rest, axis=0, equal_var= False, 
                              alternative=alt,permutations=permutations)
        
            b_dict[f'{cl} t-value, {alt}'] = statistics_b
            gamma_dict[f'{cl} t-value, {alt}'] = statistics_gamma
            mu1_dict[f'{cl} t-value, {alt}'] = statistics_mu1
            mu2_dict[f'{cl} t-value, {alt}'] = statistics_mu2
            
            b_dict[f'{cl} p-value, {alt}'] = pvalues_b
            gamma_dict[f'{cl} p-value, {alt}'] = pvalues_gamma
            mu1_dict[f'{cl} p-value, {alt}'] = pvalues_mu1
            mu2_dict[f'{cl} p-value, {alt}'] = pvalues_mu2
    if return_dataframe == True:
        return(pd.DataFrame(b_dict),pd.DataFrame(gamma_dict),pd.DataFrame(mu1_dict),pd.DataFrame(mu2_dict))
    else:
        return(b_dict,gamma_dict,mu1_dict,mu2_dict)
    
    
def perform_ranksumtest(adata,genes,results_dict,cluster_label='subclass_label',setup = 'Bursty-10-NAS_SHAPE',
                   alternative = ['two-sided'],subset_index=False, return_dataframe = False):
    ''' Performs Wilcoxon Rank Sum test on mu1 and mu2 of scVI parameters. 
    '''
    
    # subset the results_dict and adata if subset != False
    if subset_index != False:
        adata = adata[subset_index]
        norm_means = np.array(results_dict[setup]['norm_means'])[subset_index]
        dispersions = results_dict[setup]['params']['dispersions'][subset_index]
    else:
        norm_means = np.array(results_dict[setup]['norm_means'])
        dispersions = results_dict[setup]['params']['dispersions']
        
    # define dictionaries to hold t and p values
    b_dict = {}
    gamma_dict = {}
    mu1_dict = {}
    mu2_dict = {}
    
    b_dict['genes'] = genes
    gamma_dict['genes'] = genes
    mu1_dict['genes'] = genes
    mu2_dict['genes'] = genes
    
    # define cluster labels
    cluster_labels = np.array(adata.obs[cluster_label].tolist())
    unique_clusters = np.unique(cluster_labels)
    adata_genes = adata.var[gene_names].tolist()

    N = int(len(results_dict[setup]['params']['mean'][0,:])/2)
    
    # arrays to hold parameters
    b = np.ones((len(adata),len(genes)))
    gamma = np.ones((len(adata),len(genes)))
    mu1 = np.ones((len(adata),len(genes)))
    mu2 = np.ones((len(adata),len(genes)))

    
    # loop through genes and calculate parameters
    for i,g in enumerate(genes):
        index = adata_genes.index(g)
        mu1_ = norm_means[:,index]
        mu2_ =  norm_means[:,index+N]
        theta = dispersions[:,index]
    
        b_,beta_,gamma_ = get_bursty_params(mu1_,mu2_,theta)
        b[:,i] = b_
        gamma[:,i] = gamma_
        
        mu1[:,i] = mu1_
        mu2[:,i] = mu2_
            
    for cl in unique_clusters:
        print(cl)
        # subset parameters
        b_cl = b[cluster_labels == cl,:]
        gamma_cl = gamma[cluster_labels == cl,:]
        mu1_cl = mu1[cluster_labels == cl,:]
        mu2_cl = mu2[cluster_labels == cl,:]
        
        b_rest = b[cluster_labels != cl,:]
        gamma_rest = gamma[cluster_labels != cl,:]
        mu1_rest = mu1[cluster_labels != cl,:]
        mu2_rest = mu2[cluster_labels != cl,:]
        
        
        # fold change: av (normalized param in cl/normalized param in rest)
        b_dict[f'{cl} fold change'] = np.mean(b_cl,axis=0)/np.mean(b_rest,axis=0)
        gamma_dict[f'{cl} fold change'] = np.mean(gamma_cl,axis=0)/np.mean(gamma_rest,axis=0)
        mu1_dict[f'{cl} fold change'] = np.mean(mu1_cl,axis=0)/np.mean(mu1_rest,axis=0)
        mu2_dict[f'{cl} fold change'] = np.mean(mu2_cl,axis=0)/np.mean(mu2_rest,axis=0)
        
        # statistics for each parameters
        for alt in alternative:
            statistics_b, pvalues_b = ranksums(b_cl, b_rest, axis=0,
                              alternative=alt)
            statistics_gamma, pvalues_gamma = ranksums(gamma_cl, gamma_rest, axis=0,
                              alternative=alt)
            statistics_mu1, pvalues_mu1 = ranksums(mu1_cl, mu1_rest, axis=0,
                              alternative=alt)
            statistics_mu2, pvalues_mu2 = ranksums(mu2_cl, mu2_rest, axis=0,
                              alternative=alt)
        
            b_dict[f'{cl} t-value, {alt}'] = statistics_b
            gamma_dict[f'{cl} t-value, {alt}'] = statistics_gamma
            mu1_dict[f'{cl} t-value, {alt}'] = statistics_mu1
            mu2_dict[f'{cl} t-value, {alt}'] = statistics_mu2
            
            b_dict[f'{cl} p-value, {alt}'] = pvalues_b
            gamma_dict[f'{cl} p-value, {alt}'] = pvalues_gamma
            mu1_dict[f'{cl} p-value, {alt}'] = pvalues_mu1
            mu2_dict[f'{cl} p-value, {alt}'] = pvalues_mu2

    if return_dataframe == True:
        return(pd.DataFrame(b_dict),pd.DataFrame(gamma_dict),pd.DataFrame(mu1_dict),pd.DataFrame(mu2_dict))
    else:
        return(b_dict,gamma_dict,mu1_dict,mu2_dict)