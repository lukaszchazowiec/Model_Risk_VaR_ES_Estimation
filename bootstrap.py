import numpy as np

from config import Config
from simulate import simulate_garch_t
from estimate import fit_garch_t, var_es_from_params, conditional_sigma2_next


def parametric_bootstrap(returns: np.ndarray, fit_result: dict, N: int, cfg: Config, seed: int) -> dict:
    """
    Parametric bootstrap for VaR/ES uncertainty.

    Treats fit_result's parameters (alpha, beta, omega, nu) as if they were
    the true model. Simulates cfg.B new synthetic series from them, refits
    each one, and recomputes VaR/ES each time. The spread of these B
    estimates approximates how much the ORIGINAL estimate could have varied
    due to sampling error alone.
    """

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(cfg.B)

    var_samples = []
    es_samples = []
    n_failed = 0

    for i in range(cfg.B):
        synthetic = simulate_garch_t(cfg, N=N, seed=child_seeds[i],
                                     alpha=fit_result["alpha"],
                                     beta=fit_result["beta"],
                                     omega=fit_result["omega"],
                                     nu=fit_result["nu"])
        new_fit = fit_garch_t(synthetic["returns"])

        if not new_fit["converged"]:
            n_failed += 1
            continue

        sigma2_cond = conditional_sigma2_next(returns, new_fit["alpha"], new_fit["beta"], new_fit["omega"])
        replicate_var_es = var_es_from_params(sigma2_cond, new_fit["nu"], cfg)
        var_samples.append(replicate_var_es["var"])
        es_samples.append(replicate_var_es["es"])

    var_samples = np.array(var_samples)
    es_samples = np.array(es_samples)

    lower_pct = (1 - cfg.ci_level) / 2 * 100
    upper_pct = (1 - (1 - cfg.ci_level) / 2) * 100

    var_ci = np.percentile(var_samples, [lower_pct, upper_pct])
    es_ci = np.percentile(es_samples, [lower_pct, upper_pct])

    return {
        "var_ci": var_ci,
        "es_ci": es_ci,
        "var_samples": var_samples,
        "es_samples": es_samples,
        "n_failed": n_failed,
    }

def residual_bootstrap(returns: np.ndarray, fit_result: dict, N: int, cfg: Config, seed: int) -> dict:

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(cfg.B)

    var_samples = []
    es_samples = []
    n_failed = 0

    for i in range(cfg.B):
        rng = np.random.default_rng(child_seeds[i])
        centered_resid = fit_result["std_resid"] - fit_result["std_resid"].mean()
        z_star = rng.choice(centered_resid, size=N + cfg.burn_in, replace=True)
        synthetic = simulate_garch_t(cfg, N=N, seed=child_seeds[i],
                                     alpha=fit_result["alpha"],
                                     beta=fit_result["beta"],
                                     omega=fit_result["omega"],
                                     nu=fit_result["nu"],
                                     z=z_star)
        new_fit = fit_garch_t(synthetic["returns"])

        if not new_fit["converged"]:
            n_failed += 1
            continue

        sigma2_cond = conditional_sigma2_next(returns, new_fit["alpha"], new_fit["beta"], new_fit["omega"])
        replicate_var_es = var_es_from_params(sigma2_cond, new_fit["nu"], cfg)
        var_samples.append(replicate_var_es["var"])
        es_samples.append(replicate_var_es["es"])

    var_samples = np.array(var_samples)
    es_samples = np.array(es_samples)

    lower_pct = (1 - cfg.ci_level) / 2 * 100
    upper_pct = (1 - (1 - cfg.ci_level) / 2) * 100

    var_ci = np.percentile(var_samples, [lower_pct, upper_pct])
    es_ci = np.percentile(es_samples, [lower_pct, upper_pct])

    return {
        "var_ci": var_ci,
        "es_ci": es_ci,
        "var_samples": var_samples,
        "es_samples": es_samples,
        "n_failed": n_failed,
    }



if __name__ == "__main__":
    cfg = Config()
    sim_result = simulate_garch_t(cfg, N=500, seed=0)
    fit_result = fit_garch_t(sim_result["returns"])

    boot = parametric_bootstrap(sim_result["returns"], fit_result, N=500, cfg=cfg, seed=1)
    print("VaR CI:", boot["var_ci"])
    print("ES CI:", boot["es_ci"])
    print("N failed:", boot["n_failed"])

    true = var_es_from_params(sim_result["sigma2_next"], cfg.nu, cfg)
    print("true VaR/ES:", true["var"], true["es"])

    from dataclasses import replace

    cfg_small = replace(cfg, B=20)
    resid_boot = residual_bootstrap(sim_result["returns"], fit_result, N=500, cfg=cfg_small, seed=1)
    print("residual VaR CI:", resid_boot["var_ci"])
    print("residual ES CI:", resid_boot["es_ci"])
    print("N failed:", resid_boot["n_failed"])
    print(fit_result["std_resid"].mean(), fit_result["std_resid"].var())
    print(conditional_sigma2_next(sim_result["returns"], fit_result["alpha"], fit_result["beta"], fit_result["omega"]))
    print(fit_result["sigma2_next"])