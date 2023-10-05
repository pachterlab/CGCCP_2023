from typing import Union, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
from torch.distributions import constraints, Distribution, Gamma, Poisson
from torch.distributions.utils import (
    broadcast_all,
    probs_to_logits,
    lazy_property,
    logits_to_probs,
)


class BivariateNegativeBinomial(Distribution):
    # """
    # Negative binomial distribution.
    # One of the following parameterizations must be provided:
    # (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    # the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    # parameterization, which is the one used by scvi-tools. These parameters respectively
    # control the mean and inverse dispersion of the distribution.
    # In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:
    # 1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    # 2. :math:`x \sim \textrm{Poisson}(w)`
    # Parameters
    # ----------
    # total_count
    #     Number of failures until the experiment is stopped.
    # probs
    #     The success probability.
    # mu
    #     Mean of the distribution.
    # theta
    #     Inverse dispersion.
    # validate_args
    #     Raise ValueError if arguments do not match constraints
    # model_weight
    # """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        validate_args: bool = False,
        custom_dist = None,
        scale: Optional[torch.Tensor] = None,
        THETA_IS: str ='NAS_SHAPE',
        **kwargs,
    ):

        super().__init__(validate_args=validate_args)


        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu1,mu2 = torch.chunk(mu,2,dim=-1)
            mu1, mu2, theta = broadcast_all(mu1, mu2, theta)

        #### Modified for bivariate
        self.mu = mu 
        self.mu1, self.mu2 = mu1, mu2
        self.theta = theta
        self.use_custom = custom_dist is not None
        self.custom_dist = custom_dist
        self.scale = scale
        self.THETA_IS = THETA_IS
        
#         print('MEANS UNSPLICED SHAPE', mu1.shape)
#         print('MEANS UNSPLICED',mu1)
        
#         print('MEANS SPLICED SHAPE', mu2.shape)
#         print('MEANS SPLICED',mu2)
        
#         print('THETA SHAPE', theta.shape)
#         print('THETA',theta)
 
     
    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean ** 2) / self.theta

    def sample(
        self, sample_shape: Union[torch.Size, Tuple] = torch.Size()
    ) -> torch.Tensor:
        with torch.no_grad():
            gamma_d = self._gamma()
            p_means = gamma_d.sample(sample_shape)

            # Clamping as distributions objects can have buggy behaviors when
            # their parameters are too high
            l_train = torch.clamp(p_means, max=1e8)
            counts = Poisson(
                l_train
            ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
            return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )

        if self.use_custom:
            calculate_log_nb = log_prob_custom
            log_nb = calculate_log_nb(value,
                                      mu1=self.mu1, mu2=self.mu2,
                                      theta=self.theta, eps=self._eps,
                                      THETA_IS = self.THETA_IS,
                                      custom_dist = self.custom_dist)
        else:
            log_nb = log_prob_NBuncorr(value,
                                      mu1 = self.mu1, mu2 = self.mu2, eps = self._eps)
        
        return log_nb

    def _gamma(self):
        return _gamma(self.theta, self.mu)
    
def log_prob_custom(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor,  THETA_IS, eps=1e-8,
                             custom_dist=None, **kwargs):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model
    where individual genes use one of the distributions
    """

    assert custom_dist is not None, "Input a custom_dist"
    res = custom_dist(x=x, mu1=mu1, mu2=mu2, theta=theta, eps=eps,  THETA_IS = THETA_IS)

    return res



   


def log_prob_poisson(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor, THETA_IS, eps, **kwargs):
    ''' Calculates the uncorrelated Poisson likelihood for nascent and mature: just returns Poisson(n; mu1)*Poisson(m; mu2).'''
    # Divide the original data x into spliced (x) and unspliced (y)
    n,m = torch.chunk(x,2,dim=-1)

    # DOES NOT USE THETA AT ALL

    #compute the Poisson term for n and m (uncorrelated)
    y_n = n * torch.log(mu1+eps) - mu1- torch.lgamma(n+1) 
    y_m = m * torch.log(mu2+eps) - mu2- torch.lgamma(m+1) 

    P = y_n + y_m

        
    return P


def log_prob_NBcorr(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor, THETA_IS, eps=1e-8):
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
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """

    # Divide the original data x into spliced (x) and unspliced (y)
    x,y = torch.chunk(x,2,dim=-1)

    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta1.size(0)
        )  # In this case, we reshape theta for broadcasting

    log_theta_mu_eps = torch.log(theta + mu1 + mu2 + eps) # theta1 used here

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu1 + eps) - log_theta_mu_eps)
        + y * (torch.log(mu2 + eps) - log_theta_mu_eps)
        + torch.lgamma(x + y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        - torch.lgamma(y + 1)
    )

    return res

def log_prob_NBuncorr(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor, THETA_IS, eps=1e-8):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model
    where spliced and unspliced are predicted separately.
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
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """

    # Divide the original data x into spliced (x) and unspliced (y)
    x,y = torch.chunk(x,2,dim=-1)

    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta1.size(0)
        )  # In this case, we reshape theta for broadcasting

    # In contrast to log_nb_positive_bi,
    log_theta_mu1_eps = torch.log(theta + mu1 + eps)
    log_theta_mu2_eps = torch.log(theta + mu2 + eps)

    res = (
        theta * (2* torch.log(theta + eps) - log_theta_mu1_eps - log_theta_mu2_eps)
        + x * (torch.log(mu1 + eps) - log_theta_mu1_eps)
        + torch.lgamma(x + theta)
        - 2*torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + y * (torch.log(mu2 + eps) - log_theta_mu2_eps)
        + torch.lgamma(y + theta)
        - torch.lgamma(y + 1)
    )

    return res
                      

def log_prob_NBuncorr(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor, THETA_IS, eps=1e-8):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model
    where spliced and unspliced are predicted separately.
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
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """

    # Divide the original data x into spliced (m) and unspliced (n)
    # divide theta as well
    n,m = torch.chunk(x,2,dim=-1)
    theta1,theta2 = torch.chunk(theta,2,dim=-1)
    
    # In contrast to log_nb_positive_bi,
    log_theta1_mu1_eps = torch.log(theta1 + mu1 + eps)
    log_theta2_mu2_eps = torch.log(theta2 + mu2 + eps)

    
    res1 = (
        theta1 * (torch.log(theta1 + eps) - log_theta1_mu1_eps)
        + n * (torch.log(mu1 + eps) - log_theta1_mu1_eps)
        + torch.lgamma(n + theta1)
        - torch.lgamma(theta1)
        - torch.lgamma(n + 1)
    )
    
    res2 = (
        theta2 * (torch.log(theta2 + eps) - log_theta2_mu2_eps)
        + m * (torch.log(mu2 + eps) - log_theta2_mu2_eps)
        + torch.lgamma(m + theta2)
        - torch.lgamma(theta2)
        - torch.lgamma(m + 1)
    )
    return res1+res2