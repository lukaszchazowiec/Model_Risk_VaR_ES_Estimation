import math
import numpy as np

from scipy.stats import t as t_dist
from arch import arch_model

from config import Config


def fit_garch_t(returns: np.ndarray) -> dict:
    """

    :rtype: dict
    """
    model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    fit_result = model.fit(disp="off")
    forecast = fit_result.forecast(horizon=1)
    sigma2_next = forecast.variance.values[-1, 0]

    converged = fit_result.convergence_flag == 0

    return {"alpha": fit_result.params["alpha[1]"],
            "beta": fit_result.params["beta[1]"],
            "omega": fit_result.params["omega"],
            "nu": fit_result.params["nu"],
            "sigma2_next": sigma2_next,
            "std_resid": fit_result.std_resid,
            "converged": converged,
    }


def conditional_sigma2_next(returns: np.ndarray, alpha: float, beta: float, omega: float) -> float:
    sigma2 = returns.var()
    for r in returns:
        sigma2 = omega + alpha * r ** 2 + beta * sigma2

    return sigma2


def var_es_from_params(sigma2_next: float, nu: float, cfg: Config) -> dict:
    sigma_next = math.sqrt(sigma2_next)
    scale = math.sqrt((nu - 2) / nu)

    a_var = 1 - cfg.var_level
    t_a_var = t_dist.ppf(a_var, df=nu)
    var = -t_a_var * sigma_next * scale

    a_es = 1 - cfg.es_level
    t_a_es = t_dist.ppf(a_es, df=nu)
    es = sigma_next * scale * (t_dist.pdf(t_a_es, df=nu) / a_es) * ((nu + t_a_es ** 2) / (nu - 1))

    return {"var": float(var), "es": float(es)}

