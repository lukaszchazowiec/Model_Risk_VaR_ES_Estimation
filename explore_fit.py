from arch import arch_model
from config import Config
from simulate import simulate_garch_t


cfg = Config()
result = simulate_garch_t(cfg, N=500, seed=0)
returns = result["returns"]

model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
fit_result = model.fit(disp="off")
forecast = fit_result.forecast(horizon=1)
sigma2_next_est = forecast.variance.values[-1, 0]

print(fit_result.params)
print("estimated:", sigma2_next_est, "true:", result["sigma2_next"])