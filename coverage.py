import numpy as np

from config import Config
from dataclasses import replace
from bootstrap import parametric_bootstrap, residual_bootstrap, bayesian_bootstrap
from simulate import simulate_garch_t, true_var_es, skewt_rvs, skewt_true_var_es
from estimate import fit_garch_t


def coverage_metrics(N: int, cfg: Config, seed: int, R: int,
                     modes=("theoretical", "empirical"), dgp: str = "t", lam: float = 0.0):

    seed_seq = np.random.SeedSequence(seed)
    path_seeds = seed_seq.spawn(R)

    methods = {"parametric": parametric_bootstrap,
               "residual": residual_bootstrap,
               "bayesian": bayesian_bootstrap}

    # dla każdej ścieżki: [trafienie, szerokość, szerokość względna]
    paths = {f"{m} {mode} {k}": [] for m in methods for mode in modes for k in ("var", "es")}
    n_valid = 0

    for r in range(R):
        path_seed_int = int(path_seeds[r].generate_state(1)[0])
        if dgp == "skewt":
            z = skewt_rvs(cfg.burn_in + N, cfg.nu, lam, np.random.default_rng(path_seed_int))
            sim_result = simulate_garch_t(cfg, N=N, seed=path_seed_int, z=z)
            true_values = skewt_true_var_es(sim_result["sigma2_next"], cfg.nu, lam, cfg)
        else:
            sim_result = simulate_garch_t(cfg, N=N, seed=path_seed_int)
            true_values = true_var_es(sim_result["sigma2_next"], cfg)
        fit_result = fit_garch_t(sim_result["returns"])

        if not fit_result["converged"]:
            continue
        n_valid += 1

        for name, boot_fn in methods.items():
            for mode in modes:
                boot = boot_fn(sim_result["returns"], fit_result, N=N,
                               cfg=replace(cfg, quantile_mode=mode), seed=path_seed_int)
                for k in ("var", "es"):
                    low, high = boot[f"{k}_ci"]
                    paths[f"{name} {mode} {k}"].append([int(low < true_values[k] < high),
                                                        float(high - low),
                                                        float((high - low) / true_values[k])])
        if (r + 1) % 10 == 0:
            print(f"  path {r + 1}/{R}", flush=True)

    summary = {"n_valid": n_valid}
    for key, rows in paths.items():
        a = np.array(rows)
        summary[f"{key} coverage"] = float(a[:, 0].mean())
        summary[f"{key} mean width"] = float(a[:, 1].mean())
        summary[f"{key} mean rel width"] = float(a[:, 2].mean())

    return {"summary": summary, "paths": paths}


if __name__ == "__main__":
    cfg = Config()
    dev = replace(cfg, B=100)
    N = 2500
    metrics = coverage_metrics(N, dev, seed=42, R=80)

    print(metrics)