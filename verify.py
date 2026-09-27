import numpy as np
import math
from scipy.stats import t as t_dist
from config import Config
from simulate import true_var_es

if __name__ == "__main__":
    cfg = Config()
    sigma2_next = 1.0
    sigma_next = math.sqrt(sigma2_next)
    scale = math.sqrt((cfg.nu - 2) / cfg.nu)

    rng = np.random.default_rng(42)
    z = rng.standard_t(df=cfg.nu, size=10_000_000) * scale
    r_hypothetical = sigma_next * z

    a_var = 1 - cfg.var_level
    empirical_var = -np.percentile(r_hypothetical, a_var * 100)

    a_es = 1 - cfg.es_level
    threshold = np.percentile(r_hypothetical, a_es * 100)
    tail_losses = r_hypothetical[r_hypothetical <= threshold]
    empirical_es = -tail_losses.mean()

    closed_form = true_var_es(sigma2_next, cfg)
    print("VaR - empirical:", empirical_var, "closed-form:", closed_form["var"])
    print("ES - empirical:", empirical_es, "closed-form:", closed_form["es"])