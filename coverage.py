import numpy as np

from config import Config
from dataclasses import replace
from bootstrap import parametric_bootstrap, residual_bootstrap
from simulate import simulate_garch_t, true_var_es
from estimate import fit_garch_t


def coverage_metrics(N: int, cfg: Config, seed: int, R: int):

    seed_seq = np.random.SeedSequence(seed)
    path_seeds = seed_seq.spawn(R)

    param_hits_var = 0
    param_hits_es = 0
    resid_hits_var = 0
    resid_hits_es = 0

    for r in range(R):
        path_seed_int = path_seeds[r].generate_state(1)[0]
        sim_result = simulate_garch_t(cfg, N=N, seed=path_seed_int)
        fit_result = fit_garch_t(sim_result["returns"])

        if not fit_result["converged"]:
            continue
        true_values = true_var_es(sim_result["sigma2_next"], cfg)

        # PARAMETRIC VAR/ES COVERAGE
        param_boot = parametric_bootstrap(sim_result["returns"], fit_result, N=N, cfg=cfg, seed=path_seed_int)
        param_low_var, param_high_var = param_boot["var_ci"]

        if param_low_var < true_values["var"] < param_high_var:
            param_hits_var += 1

        param_low_es, param_high_es = param_boot["es_ci"]
        if param_low_es < true_values["es"] < param_high_es:
            param_hits_es += 1

        # RESIDUAL VAR/ES COVERAGE
        resid_boot = residual_bootstrap(sim_result["returns"], fit_result, N=N, cfg=cfg, seed=path_seed_int)
        resid_low_var, resid_high_var = resid_boot["var_ci"]

        if resid_low_var < true_values["var"] < resid_high_var:
            resid_hits_var += 1

        resid_low_es, resid_high_es = resid_boot["es_ci"]
        if resid_low_es < true_values["es"] < resid_high_es:
            resid_hits_es += 1


    return {"parametric var coverage": param_hits_var / R, "parametric es coverage": param_hits_es / R,
            "residual var coverage": resid_hits_var / R, "residual es coverage": resid_hits_es / R}


if __name__ == "__main__":
    cfg = Config()
    dev = replace(cfg, B=100)
    N = 2500
    metrics = coverage_metrics(N, dev, seed=42, R=80)

    print(metrics)