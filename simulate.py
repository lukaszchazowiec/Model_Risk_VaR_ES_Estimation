from scipy.special import gammaln
from scipy.stats import t as t_dist
import math
import numpy as np

from config import Config
from estimate import var_es_from_params


def simulate_garch_t(cfg: Config, N: int, seed: int,
                     alpha: float=None, beta: float=None,
                     omega: float=None, nu: float=None, z: np.ndarray = None) -> dict:
    """
    Generate one GARCH(1,1)-t path.

    By default, uses the TRUE parameters stored in cfg (alpha, beta, omega, nu).
    Pass alpha/beta/omega/nu explicitly to simulate from a DIFFERENT set of
    parameters instead (e.g., fitted parameters, for the parametric bootstrap),
    while still using cfg for settings that don't change: burn_in, target_var
    (only used as the fallback/default omega source).
    """

    # fall back to cfg's true values only when an override wasn't given
    alpha = cfg.alpha if alpha is None else alpha
    beta  = cfg.beta  if beta  is None else beta
    omega = cfg.omega if omega is None else omega
    nu    = cfg.nu    if nu    is None else nu

    total_steps = cfg.burn_in + N

    if z is None:
        rng = np.random.default_rng(seed)
        z_raw = rng.standard_t(df=nu, size=total_steps)
        z = z_raw * math.sqrt((nu - 2) / nu)
    elif len(z) != total_steps:
        raise ValueError(f"z must be of length burn_in + N = {total_steps}, got {len(z)}" )

    sigma2 = np.zeros(total_steps)
    sigma2[0] = cfg.target_var

    r = np.zeros(total_steps)
    for t in range(1, total_steps):
        r[t - 1] = math.sqrt(sigma2[t-1]) * z[t-1]
        sigma2[t] = omega + alpha * r[t-1] ** 2 + beta * sigma2[t-1]

    r[-1] = math.sqrt(sigma2[-1]) * z[-1]

    sigma2_next = omega + alpha * r[-1] ** 2 + beta * sigma2[-1]

    r_kept = r[cfg.burn_in:]

    return {"returns": r_kept, "sigma2_next": sigma2_next}


def true_var_es(sigma2_next: float, cfg: Config) -> dict:
    return var_es_from_params(sigma2_next, cfg.nu, cfg)


def _skewt_consts(nu: float, lam: float):
    c = math.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam ** 2 - a ** 2)
    s = math.sqrt(nu / (nu - 2))
    return a, b, s


def skewt_ppf(u, nu: float, lam: float) -> np.ndarray:
    """Quantile function of Hansen's skewed t (zero mean, unit variance)."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    a, b, s = _skewt_consts(nu, lam)
    left = u < (1 - lam) / 2
    z = np.empty_like(u)
    z[left] = ((1 - lam) / s * t_dist.ppf(u[left] / (1 - lam), df=nu) - a) / b
    z[~left] = ((1 + lam) / s * t_dist.ppf((u[~left] - (1 - lam) / 2) / (1 + lam) + 0.5, df=nu) - a) / b
    return z


def skewt_rvs(size: int, nu: float, lam: float, rng: np.random.Generator) -> np.ndarray:
    u = np.clip(rng.uniform(size=size), 1e-12, 1 - 1e-12)
    return skewt_ppf(u, nu, lam)


def skewt_var_es_unit(nu: float, lam: float, var_level: float, es_level: float):
    """VaR and ES (positive losses) of a unit-variance skew-t innovation."""
    a, b, s = _skewt_consts(nu, lam)
    alpha_es = 1 - es_level
    var = -float(skewt_ppf([1 - var_level], nu, lam)[0])
    q = t_dist.ppf(alpha_es / (1 - lam), df=nu)
    es = (((1 - lam) ** 2 / s) * t_dist.pdf(q, df=nu) * (nu + q ** 2) / (nu - 1) + a * alpha_es) / (b * alpha_es)
    return var, float(es)


def skewt_true_var_es(sigma2_next: float, nu: float, lam: float, cfg: Config) -> dict:
    var_u, es_u = skewt_var_es_unit(nu, lam, cfg.var_level, cfg.es_level)
    sigma = math.sqrt(sigma2_next)
    return {"var": sigma * var_u, "es": sigma * es_u}


if __name__ == "__main__":
    from config import DEV

    result = simulate_garch_t(DEV, N=500, seed=1)
    print(result["returns"].shape)
    print(result["sigma2_next"])
    print(true_var_es(result["sigma2_next"], DEV))

    result_override = simulate_garch_t(DEV, N=500, seed=1,
                                        alpha=0.09, beta=0.88, omega=0.02, nu=6.5)
    print(result_override["sigma2_next"])