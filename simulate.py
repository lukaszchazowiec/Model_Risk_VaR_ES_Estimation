import math
import numpy as np

from config import Config
from estimate import var_es_from_params


def simulate_garch_t(cfg: Config, N: int, seed: int):
    rng = np.random.default_rng(seed)
    total_steps = cfg.burn_in + N

    z_raw = rng.standard_t(df=cfg.nu, size=10000000)
    z = z_raw * math.sqrt((cfg.nu - 2) / cfg.nu)

    sigma2 = np.zeros(total_steps)
    sigma2[0] = cfg.target_var
    r = np.zeros(total_steps)

    for t in range(1, total_steps):
        r[t-1] = math.sqrt(sigma2[t-1]) * z[t-1]
        sigma2[t] = cfg.omega + cfg.alpha * r[t-1]**2 + cfg.beta * sigma2[t-1]
    r[-1] = math.sqrt(sigma2[-1]) * z[-1]

    sigma2_next = cfg.omega + cfg.alpha * r[-1]**2 + cfg.beta * sigma2[-1]
    r_kept = r[cfg.burn_in:]

    return {"returns": r_kept, "sigma2_next": sigma2_next}


def true_var_es(sigma2_next: float, cfg: Config) -> dict:
    return var_es_from_params(sigma2_next, cfg.nu, cfg)


if __name__ == "__main__":
    result = true_var_es(sigma2_next=1, cfg=Config())
    print(result)