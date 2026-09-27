from arch import arch_model
from config import Config
from simulate import simulate_garch_t
from estimate import fit_garch_t, var_es_from_params


cfg = Config()
result = simulate_garch_t(cfg, N=500, seed=0)
returns = result["returns"]

fit_result = fit_garch_t(returns)

truth = var_es_from_params(fit_result["sigma2_next"], cfg.nu, cfg)
estimate = var_es_from_params(result["sigma2_next"], cfg.nu, cfg)

print("true VaR/ES:      ", truth)
print("estimates VaR/ES: ", estimate)