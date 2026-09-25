"""
Conventions for this project:
mean = 0
"""

import math
import warnings
from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class Config:
    # -----model-----
    alpha: float = 0.08         # reaction to new shocks
    beta: float = 0.90          # volatility persistence
    nu: float = 6.0             # degrees of freedom

    # -----simulation-----
    seed: int = 42              # random seed
    target_var: float = 1.0     # variance in the simulation
    R: int = 500                # number of paths
    B: int = 999                # number of bootstrap samples
    burn_in: int = 500          # extra observations to discard
    sample_sizes: tuple[int, ...] = (500, 1000, 2500)

    # -----risk metrics-----
    var_level: float = 0.99     # VaR and ES values according to BASEL II and BASEL III regulations
    es_level: float = 0.975
    ci_level: float = 0.95      # confidence interval level


    # -----safety checks-----
    def __post_init__(self):

        rules = [
            (self.alpha < 0 or self.beta < 0, "alpha and beta must be non-negative"),
            (self.alpha + self.beta >= 1, "alpha + beta must be below 1, otherwise the variance will not mean-revert"),
            (self.target_var <= 0, "target_var must be positive"),
            (self.nu <= 2, "nu must be greater than 2"),
            (self.R <= 0 or self.B <= 0, "R and B must be positive"),
            (self.burn_in < 0, "burn_in cannot be negative"),
            (not 0 < self.var_level < 1, "var_level must be strictly between 0 and 1"),
            (not 0 < self.es_level < 1, "es_level must be strictly between 0 and 1"),
            (not 0 < self.ci_level < 1, "ci_level must be strictly between 0 and 1"),
            (len(self.sample_sizes) == 0 or min(self.sample_sizes) <= 0, "sample_sizes must be a non-empty tuple"
                                                                         "of positive integers"),
        ]
        for rule, message in rules:
            if rule:
                raise ValueError(message)

        if self.nu <= 4:
            warnings.warn("nu <= 4: kurtosis is infinite, cannot compute kurtosis")


    @property
    def omega(self) -> float:
        return  self.target_var * (1 - self.alpha - self.beta)

    @property
    def half_life(self) -> float:
        return math.log(0.5) / math.log(self.alpha + self.beta)


    def to_dict(self) -> dict:
        return {**asdict(self), "omega": self.omega}


FULL = Config()
DEV = replace(FULL, R=5, B=20)

if __name__ == "__main__":
    print(Config().omega)