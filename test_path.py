from scipy.stats import kurtosis

from config import Config
from simulate import simulate_garch_t


if __name__ == "__main__":
    cfg = Config()

    # Theoretical kurtosis of r_t under this GARCH-t(nu=6) process is ~15.6,
    # derived from: k_r = 3*(1-(a+b)^2) / (1-(a+b)^2 - a^2*(k_z-1))
    # with k_z = 3 + 6/(nu-4) = 6 for nu=6.
    # The 4th-moment stationarity margin is tiny (0.992), so sample kurtosis
    # converges very slowly and stays biased low at finite N. At N=500,000
    # across 15 seeds, the observed range was ~9.8-19.2 (mean ~12.8) - consistent
    # with slow convergence.
    for s in range(15):
        result = simulate_garch_t(cfg, N=500_000, seed=s)
        r = result["returns"]
        print(s, "var:", r.var(), " kurt:", kurtosis(r, fisher=False))