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


if __name__ == "__main__":
    from config import DEV

    result = simulate_garch_t(DEV, N=500, seed=1)
    print(result["returns"].shape)
    print(result["sigma2_next"])
    print(true_var_es(result["sigma2_next"], DEV))

    result_override = simulate_garch_t(DEV, N=500, seed=1,
                                        alpha=0.09, beta=0.88, omega=0.02, nu=6.5)
    print(result_override["sigma2_next"])