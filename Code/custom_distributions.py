# module with custom distributions

import torch 
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import nnNB_module as nnNB_module


def log_prob_MMNB(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor,  eps, THETA_IS, **kwargs):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model.
    Parameters
    ----------
    x
        data
    mu1,mu2
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars/2)
    theta
        params (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """

    # Divide the original data x into spliced (x) and unspliced (y)
    n,m = torch.chunk(x,2,dim=-1)

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

    var1 = mu1 * (1+b)
    var2 = mu2 * (1+b*beta/(beta+gamma))
    cov = b**2/(beta+gamma)
    
    logvar1 = torch.log((var1/mu1**2)+1)
    logvar2 = torch.log((var2/mu2**2)+1)
    logstd1 = torch.sqrt(logvar1)
    logstd2 = torch.sqrt(logvar2)

    logmean1 = torch.log(mu1**2/torch.sqrt(var1+mu1**2))
    logmean2 = torch.log(mu2**2/torch.sqrt(var2+mu2**2))

    b = (logmean1 + logmean2 + (logvar1 + logvar2)/2)
    b[b<-88] = -88
    logcov = torch.log(cov * torch.exp(-(b)) +1 )
    logcorr = logcov/torch.sqrt(logvar1 * logvar2)


    logmean_cond = logmean2 + logcorr * logstd2/logstd1 * (torch.log(n+1) - logmean1)
    logvar_cond = logvar2 * (1-logcorr**2)  
    # logstd_cond = logstd2 * torch.sqrt(1-logcorr**2)   
    # logvar_cond = logstd_cond**2


    mean_cond = torch.exp(logmean_cond + logvar_cond/2)
    var_cond = torch.exp(2*logmean_cond + logvar_cond) * (torch.exp(logvar_cond) - 1)

    r = 1/beta
    r_cond = mean_cond**2/(var_cond-mean_cond)
    p_cond = mean_cond/var_cond

    # negative binomial of nascent RNA n
    prefactor = torch.lgamma(n+r) - torch.lgamma(n+1) - torch.lgamma(r) \
                + r * torch.log(r/(r+mu1)+eps) + n * torch.log(mu1/(r+mu1)+eps)


    filt = torch.logical_and(torch.logical_and(r>0,p_cond>0), p_cond<1)

    #compute the Poisson mean
    y_ = m * torch.log(mean_cond+eps) - mean_cond - torch.lgamma(m+1) 

    y_[filt] += torch.lgamma(m[filt]+r_cond[filt]) - torch.lgamma(r_cond[filt]) \
                + r_cond[filt] * torch.log(r_cond[filt]/(r_cond[filt]+mean_cond[filt])+eps) \
                - m[filt] * torch.log(r_cond[filt]+mean_cond[filt]+eps) + mean_cond[filt]

    P = prefactor +  y_
    return P


def log_prob_poisson(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor, eps, **kwargs):
    ''' Calculates the uncorrelated Poisson likelihood for nascent and mature: just returns Poisson(n; mu1)*Poisson(m; mu2).'''
    # Divide the original data x into spliced (x) and unspliced (y)
    n,m = torch.chunk(x,2,dim=-1)

    # DOES NOT USE THETA AT ALL

    #compute the Poisson term for n and m (uncorrelated)
    y_n = n * torch.log(mu1+eps) - mu1- torch.lgamma(n+1) 
    y_m = m * torch.log(mu2+eps) - mu2- torch.lgamma(m+1) 

    P = y_n + y_m

        
    return P


log_prob_nnNB = nnNB_module.log_prob_nnNB
