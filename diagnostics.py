import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import Config
from simulate import simulate_garch_t


def fit_for_diagnostics(returns: np.ndarray) -> np.ndarray:
    """Local fit helper for this diagnostic script only — uses a scale
    factor suited to longer series (N=5000), separate from estimate.py's
    fit_garch_t (tuned for N in {500, 1000, 2500})."""
    scale_factor = 10.0
    model = arch_model(returns * scale_factor, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    fit_result = model.fit(disp="off")
    return fit_result.std_resid


if __name__ == "__main__":
    cfg = Config()
    std_resids = []

    for r in range(500):
        returns = simulate_garch_t(cfg, N=5000, seed=r)["returns"]
        std_resids.append(fit_for_diagnostics(returns))

    std_resids = np.array(std_resids)

    box_results = [acorr_ljungbox(s, lags=[10], return_df=True)["lb_pvalue"].iloc[0] for s in std_resids]
    box_results_sq = [acorr_ljungbox(s ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0] for s in std_resids]

    print(len([p for p in box_results if p < 0.05]) / len(box_results))
    print(len([p for p in box_results_sq if p < 0.05]) / len(box_results_sq))