import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox

from config import Config
from simulate import simulate_garch_t
from estimate import fit_garch_t



if __name__ == "__main__":
    cfg = Config()

    std_resids = []
    result = []

    for r in range(500):
        result = simulate_garch_t(cfg, N=5000, seed=r)["returns"]
        fit_result = fit_garch_t(result)
        std_resids.append(fit_result["std_resid"])

    std_resids = np.array(std_resids)

    box_results = []
    for s in std_resids:
        box_results.append(acorr_ljungbox(s, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

    box_results_sq = []
    for s in std_resids:
        box_results_sq.append(acorr_ljungbox(s ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

    print(len([box for box in box_results if box < 0.05]) / len(box_results))
    print(len([box for box in box_results_sq if box < 0.05]) / len(box_results_sq))