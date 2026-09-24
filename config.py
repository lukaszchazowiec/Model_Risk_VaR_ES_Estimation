import warnings
from dataclasses import dataclass, replace, asdict

@dataclass(frozen=True)
class Config:

    seed: int = 42
    alpha: float = 0.08         # reaction to new shocks
    beta: float = 0.90          # volatility persistence
    target_var: float = 1.0     # variance in the simulation
    nu: float = 6               # degrees of freedom

    # -----safety checks-----
    def __post_init__(self):

        rules = [
            (self.alpha < 0 or self.beta < 0, "alpha and beta must be non-negative!"),
            (self.alpha + self.beta >= 1, "alpha + beta cannot be more than 1, otherwise the variance will not mean-revert!"),
            (self.target_var <= 0, "target_var must be positive!"),
            #(not 0 < self.var_level < 1, "var_level must be strictly between 0 and 1"),
            (self.nu <= 0, "nu must be greater than 2!"),
        ]
        for rule, message in rules:
            if rule:
                raise ValueError(message)

        if self.nu <= 4:
            warnings.warn("nu <= 4: v is too low, cannot compute kurtosis")


    @property
    def omega(self) -> float:
        return  self.target_var * (1 - self.alpha - self.beta)


cfg = Config()
print(cfg.omega)
