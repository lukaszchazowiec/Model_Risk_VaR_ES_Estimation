import math
import numpy as np
from scipy.stats import t as tdist

from config import Config

def simulate_garch_t(cfg: Config, N: int, seed: int):
    rng = np.random.default_rng(seed)
    total_steps = cfg.burn_in + N

    z_raw = rng.standard_t(df=cfg.nu, size=total_steps)
    z = z_raw * math.sqrt((cfg.nu - 2) / cfg.nu)

    sigma2 = np.zeros(total_steps)
    sigma2[0] = cfg.target_var

    for t in range(1, total_steps):
        r_prev = math.sqrt(sigma2[t-1]) * z[t-1]
        sigma2[t] = cfg.omega + cfg.alpha * r_prev**2 + cfg.beta * sigma2[t-1]
    pass