import numpy as np

from config import Config
from bootstrap import parametric_bootstrap, residual_bootstrap
from simulate import simulate_garch_t, true_var_es
from estimate import fit_garch_t


def run_single_path(N: int, cfg: Config, path_seed_int: int) -> dict:
    sim_result = simulate_garch_t(cfg, N=N, seed=path_seed_int)
    fit_result = fit_garch_t(sim_result["returns"])

    if not fit_result["converged"]:
        return {"converged": False}

    true_values = true_var_es(sim_result["sigma2_next"], cfg)

    param_boot = parametric_bootstrap(sim_result["returns"], fit_result, N=N, cfg=cfg, seed=path_seed_int)
    resid_boot = residual_bootstrap(sim_result["returns"], fit_result, N=N, cfg=cfg, seed=path_seed_int)

    param_low_var, param_high_var = param_boot["var_ci"]
    param_low_es, param_high_es = param_boot["es_ci"]
    resid_low_var, resid_high_var = resid_boot["var_ci"]
    resid_low_es, resid_high_es = resid_boot["es_ci"]

    return {
        "converged": True,
        "true_var": true_values["var"],
        "true_es": true_values["es"],
        "param_var_ci": param_boot["var_ci"].tolist(),
        "param_es_ci": param_boot["es_ci"].tolist(),
        "resid_var_ci": resid_boot["var_ci"].tolist(),
        "resid_es_ci": resid_boot["es_ci"].tolist(),
        "param_var_hit": bool(param_low_var < true_values["var"] < param_high_var),
        "param_es_hit": bool(param_low_es < true_values["es"] < param_high_es),
        "resid_var_hit": bool(resid_low_var < true_values["var"] < resid_high_var),
        "resid_es_hit": bool(resid_low_es < true_values["es"] < resid_high_es),
    }


def coverage_metrics(N: int, cfg: Config, seed: int, R: int) -> dict:
    seed_seq = np.random.SeedSequence(seed)
    path_seeds = [s.generate_state(1)[0] for s in seed_seq.spawn(R)]

    raw_results = [run_single_path(N, cfg, s) for s in path_seeds]

    converged_results = [r for r in raw_results if r["converged"]]
    n_converged = len(converged_results)

    summary = {
        "n_paths": R,
        "n_converged": n_converged,
        "n_failed": R - n_converged,
        "parametric_var_coverage": sum(r["param_var_hit"] for r in converged_results) / n_converged,
        "parametric_es_coverage": sum(r["param_es_hit"] for r in converged_results) / n_converged,
        "residual_var_coverage": sum(r["resid_var_hit"] for r in converged_results) / n_converged,
        "residual_es_coverage": sum(r["resid_es_hit"] for r in converged_results) / n_converged,
    }

    return {"summary": summary, "raw": raw_results}


if __name__ == "__main__":
    import json

    cfg = Config()

    result = run_single_path(N=500, cfg=cfg, path_seed_int=0)
    assert result["true_es"] > result["true_var"]
    assert result["param_var_ci"][0] < result["param_var_ci"][1]
    print("run_single_path OK:", result)

    metrics = coverage_metrics(N=500, cfg=cfg, seed=42, R=10)
    assert len(metrics["raw"]) == 10
    assert all(0 <= v <= 1 for v in metrics["summary"].values() if isinstance(v, float))
    print("coverage_metrics OK:", metrics["summary"])

    json.dumps(metrics)
    print("JSON OK")