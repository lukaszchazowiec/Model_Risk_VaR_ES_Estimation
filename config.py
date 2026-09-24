import warnings
from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class Config:
    # -----model-----
    alpha: float = 0.08         # reaction to new shocks
    beta: float = 0.90          # volatility persistence
    nu: float = 6               # degrees of freedom

    # -----simulation-----
    seed: int = 42              # random seed
    target_var: float = 1.0     # variance in the simulation
    R: int = 500                # number of paths
    B: int = 999                # number of bootstrap samples
    sample_sizes: tuple[int, ...] = (500, 1000, 2500)

    # -----risk metrics-----
    var_level: float = 0.99     # VaR and ES values according to BASEL II and BASEL III
    es_level: float = 0.975
    ci_level: float = 0.95      # confidence interval level


    # -----safety checks-----
    def __post_init__(self):

        rules = [
            (self.alpha < 0 or self.beta < 0, "alpha and beta must be non-negative"),
            (self.alpha + self.beta >= 1, "alpha + beta must be below 1, otherwise the variance will not mean-revert"),
            (self.target_var <= 0, "target_var must be positive"),
            (self.nu <= 2, "nu must be greater than 2"),
            (not 0 < self.var_level < 1, "var_level must be strictly between 0 and 1"),
            (not 0 < self.es_level < 1, "es_level must be strictly between 0 and 1"),
            (not 0 < self.ci_level < 1, "ci_level must be strictly between 0 and 1"),
        ]
        for rule, message in rules:
            if rule:
                raise ValueError(message)

        if self.nu <= 4:
            warnings.warn("nu <= 4: v is too low, cannot compute kurtosis")


    @property
    def omega(self) -> float:
        return  self.target_var * (1 - self.alpha - self.beta)


FULL = Config()
DEV = replace(FULL, R=5, B=20)
