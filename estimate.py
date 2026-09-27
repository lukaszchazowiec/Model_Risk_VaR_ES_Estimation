import numpy as np
from arch import arch_model


def fit_garch_t(returns: np.ndarray) -> dict:

    model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    fit_result = model.fit(disp="off")
    forecast = fit_result.forecast(horizon=1)
    sigma2_next = forecast.variance.values[-1, 0]

    return {"alpha": fit_result.params["alpha[1]"],
            "beta": fit_result.params["beta[1]"],
            "omega": fit_result.params["omega"],
            "nu": fit_result.params["nu"],
            "sigma2_next": sigma2_next
    }

