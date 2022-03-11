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

def log_nb_positive_bi_custom(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor, mw: torch.tensor, eps=1e-8, ):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model
    where individual genes use one of the distributions
    """
    # Process the model weights with softmax
    mw = mw.view(-1,int(mw.shape[-1]/2),2)
    mw = F.softmax(mw/T,dim=-1)
    mw1,mw2=torch.chunk(mw,2,dim=-1)
    mw1 = mw1.view(-1,mw1.shape[-2])
    mw2 = mw2.view(-1,mw2.shape[-2])
    # res calculation
    res1 = log_nb_positive_bi(x,
                              mu1=mu1, mu2=mu2,
                              theta=theta, eps=eps)
    res2 = log_nb_positive_bi_uncor(x,
                                    mu1=mu1, mu2=mu2,
                                    theta=theta, eps=eps)
    res = res1 * mw1 + res2 * mw2

    return res


def log_nb_positive_bi_mixed(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor, mw: torch.tensor, eps=1e-8, T=1):
    """
    Log likelihood (scalar) of a minibatch according to a bivariate nb model
    where individual genes use one of the distributions
    """
    # Process the model weights with softmax
    mw = mw.view(-1,int(mw.shape[-1]/2),2)
    mw = F.softmax(mw/T,dim=-1)
    mw1,mw2=torch.chunk(mw,2,dim=-1)
    mw1 = mw1.view(-1,mw1.shape[-2])
    mw2 = mw2.view(-1,mw2.shape[-2])
    # res calculation
    res1 = log_nb_positive_bi(x,
                              mu1=mu1, mu2=mu2,
                              theta=theta, eps=eps)
    res2 = log_nb_positive_bi_uncor(x,
                                    mu1=mu1, mu2=mu2,
                                    theta=theta, eps=eps)
    res = res1 * mw1 + res2 * mw2

    return res

def log_nb_positive_bi(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor, eps=1e-8):
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

def log_nb_positive_bi_uncor(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                             theta: torch.Tensor, eps=1e-8):
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


class BivariateNegativeBinomial(Distribution):
    r"""
    Negative binomial distribution.
    One of the following parameterizations must be provided:
    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.
    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:
    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    validate_args
        Raise ValueError if arguments do not match constraints
    use_corr
        Boolean to select either correlated or uncorrelated nb distribution
    model_weight

    """

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
        use_corr: bool = True,
        use_mixed: bool = False,
        T: float = 1,
        custom_dist = None,
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
            if use_mixed:
                # Split the theta into three parts, spliced, unspliced, and
                # model weights (mw) for each gene
                mu,mw=torch.chunk(mu,2,dim=-1)
                mu1,mu2=torch.chunk(mu,2,dim=-1)
                mu1, mu2, theta = broadcast_all(mu1, mu2, theta)
            else:
                # Split the mu into two parts, mu1=spliced , mu2=unspliced
                mu1,mu2 = torch.chunk(mu,2,dim=-1)
                mu1, mu2, theta = broadcast_all(mu1, mu2, theta)
                mw=None

        #### Modified for bivariate
        self.mu, self.mu2 = mu1, mu2
        self.theta = theta
        self.mw = mw
        self.T=T

        self.use_corr  = use_corr
        self.use_mixed = use_mixed
        self.use_custom = custom_dist is not None
        self.custom_dist = custom_dist

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

        if self.use_mixed:
            calculate_log_nb = log_nb_positive_bi_mixed
            log_nb = calculate_log_nb(value,
                                      mu1=self.mu, mu2=self.mu2,
                                      theta=self.theta, eps=self._eps,
                                      mw=self.mw, T=self.T)
        elif self.use_custom:
            calculate_log_nb = log_nb_positive_bi_custom
            log_nb = calculate_log_nb(value,
                                      mu1=self.mu, mu2=self.mu2,
                                      theta=self.theta, eps=self._eps,
                                      mw=self.mw, T=self.T,
                                      custom_dist = self.custom_dist)
        else:
            if self.use_corr:
                calculate_log_nb = log_nb_positive_bi
            else:
                calculate_log_nb = log_nb_positive_bi_uncor
            log_nb = calculate_log_nb(value,
                                      mu1=self.mu, mu2=self.mu2,
                                      theta=self.theta, eps=self._eps)

        return log_nb

    def _gamma(self):
        return _gamma(self.theta, self.mu)
