def log_nb_positive_bi(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor,
                       theta: torch.Tensor, model: torch.Tensor, eps=1e-8):
    res = torch.empty_like(mu1)
    s=model
    res[:,s] = log_nb_bivariate_correlated(x[:,s], mu1[:,s], mu2[:,s], theta[s], eps)
    not_s = tf.math.logical_not(s)
    res[:,not_s] = log_nb_bivariate_uncorrelated(x[:,not_s], mu1[:,not_s], mu2[:,not_s], theta[not_s], eps)
    return res

class BivariateNegativeBinomial(Distribution):
    r"""
    Parameters
    ----------
    x
        Data (nascent + mature concatenated)
    mu1
        Mean of the nascent distribution.
    mu2
        Mean of the mature distribution.
    theta
        Inverse dispersion.
    model
        The model index (boolean tensor the same size as theta -- each gene has a particular model)
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "model": constraints.boolean #not sure this works for tensors
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        model: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if not using_param_1:
            mu1,mu2 = torch.chunk(mu,2,dim=-1) # Split the theta into two parts
            mu1, mu2, theta, model = broadcast_all(mu1, mu2, theta, model)
        else:
            raise ValueError('use the other parametrization')

        #### Modified for bivariate with multiple models
        self.mu, self.mu2 = mu1, mu2
        self.theta = theta
        self.model = model

        super().__init__(validate_args=validate_args)


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ### ...
        return log_nb_positive_bi(value, mu1=self.mu, mu2=self.mu2, theta=self.theta, model=self.model, eps=self._eps)
