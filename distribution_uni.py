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

def log_nb_positive_uni_custom(x: torch.Tensor, mu: torch.Tensor,
                             theta: torch.Tensor, eps=1e-8,
                             custom_dist=None, **kwargs):
    """
    Log likelihood (scalar) of a minibatch according to an nb model
    where individual genes use one of the distributions
    """

    assert custom_dist is not None, "Input a custom_dist"
    res = custom_dist(x=x, mu=mu, theta=theta, eps=eps)

    return res


def log_nb_positive_uni_mixed(x: torch.Tensor, mu: torch.Tensor,
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
    res1 = log_nb_positive_uni(x,
                              mu=mu,
                              theta=theta, eps=eps)
    res2 = log_nb_positive_uni_custom(x,
                                    mu=mu,
                                    theta=theta, eps=eps)
    res = res1 * mw1 + res2 * mw2

    return res

def log_nb_positive_uni(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """
    Log likelihood (scalar) of a minibatch according to an nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res



class UnivariateNegativeBinomial(Distribution):
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
        mode: str = 'vanilla',
        validate_args: bool = False,
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
                print('ASJDHAKSJDHAKSJDHAKSJDh')

            else:
                mu = mu
                mw = None

        #### Modified 
        self.mu = mu
        self.theta = theta
        self.mw = mw
        self.T=T
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

        if self.use_custom:
            # removed the mw from custom distribution
            calculate_log_nb = log_nb_positive_uni_custom
            log_nb = calculate_log_nb(value,
                                      mu=self.mu, 
                                      theta=self.theta, eps=self._eps,
                                      T=self.T,
                                      custom_dist = self.custom_dist)
        elif self.use_mixed:
            calculate_log_nb = log_nb_positive_uni_mixed
            log_nb = calculate_log_nb(value,
                                      mu=self.mu,
                                      theta=self.theta, eps=self._eps,
                                      mw=self.mw, T=self.T)
        else:
            calculate_log_nb = log_nb_positive_uni
            log_nb = calculate_log_nb(value,
                                      mu=self.mu,
                                      theta=self.theta, eps=self._eps)

        return log_nb

    def _gamma(self):
        return _gamma(self.theta, self.mu)