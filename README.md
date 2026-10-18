# Model Risk in VaR/ES Estimation — Bootstrap Confidence Intervals for GARCH(1,1)-t

## 1. Overview

This project investigates **model risk in one-day-ahead VaR and Expected Shortfall (ES)
forecasts** produced by a GARCH(1,1) model with Student-*t* innovations. The central
question: when a GARCH-t model is fitted to a finite sample and used to forecast VaR/ES,
how much uncertainty does that estimate actually carry — and do standard bootstrap
confidence intervals honestly reflect that uncertainty?

The project is in two parts:

- **Part 1 — Simulation pilot (this README).** A fully controlled Monte Carlo study
  where the true data-generating process is known exactly, so the true VaR/ES can be
  computed in closed form and compared against bootstrap-based estimates.
- **Part 2 — Empirical application (to be added).** Calibration to real index data
  (WIG20 / S&P 500) and application of the same methodology to real returns.

This document currently covers Part 1 in full.

## 2. Methodology

### 2.1 Data-generating process

Returns are simulated from a GARCH(1,1) model with standardized Student-*t* innovations:

```
r_t = σ_t · z_t
σ_t² = ω + α·r²_{t-1} + β·σ²_{t-1}
z_t ~ standardized Student-t(ν), unit variance
```

**True parameters used in the pilot:** α = 0.08, β = 0.90, ν = 6.0, target
(unconditional) variance = 1.0 → ω ≈ 0.02. Mean fixed at μ = 0 (standard simplification
at the one-day horizon, where the mean is negligible relative to volatility and
difficult to estimate reliably at any realistic sample size). Returns in percent units.
Losses reported as positive numbers throughout.

These parameters were chosen to resemble typical daily-equity GARCH fits (persistence
α+β = 0.98, implying a volatility half-life of ≈34 days) but were **not** calibrated to
real data for this pilot — see Section 6 for the planned WIG20/S&P 500 calibration.

### 2.2 True VaR/ES (closed form)

Because the true parameters are known, σ²_{T+1} is observable exactly given the
simulated history, and the conditional distribution of r_{T+1} is a scaled Student-*t*.
True VaR and ES therefore have closed-form expressions in terms of σ_{T+1}, ν, and the
t-distribution's quantile/density functions — no estimation error. This is the "ground
truth" every bootstrap interval is checked against.

Risk levels: VaR at 99%, ES at 97.5% (Basel/FRTB convention).

### 2.3 Estimation

For each simulated path, a GARCH(1,1)-t model is refit by maximum likelihood (`arch`
package), yielding estimated α̂, β̂, ω̂, ν̂ and a one-step-ahead forecast σ̂²_{T+1}. The
same closed-form VaR/ES function used for the truth is reused for the estimate (same
formula, different inputs), so any gap between true and estimated VaR/ES is attributable
only to estimation error, not to a difference in methodology.

### 2.4 Bootstrap confidence intervals

Two bootstrap methods are implemented, both producing a 95% CI around the estimated
VaR/ES via the percentile method (B replicates, 2.5th/97.5th percentiles):

- **Parametric bootstrap.** B synthetic series are generated using the *fitted*
  parameters as if they were true, with fresh Student-*t* draws. Each is refit, and
  VaR/ES recomputed.
- **Residual bootstrap** (Pascual, Romo & Ruiz, 2006, *Computational Statistics & Data
  Analysis*). B synthetic series are generated using the fitted parameters, with shocks
  resampled with replacement from the (centered) standardized residuals of the original
  fit, rather than drawn from a parametric distribution.

**Critical detail (both methods):** following Pascual et al., the one-step-ahead
forecast used for each bootstrap replicate's VaR/ES is computed by running the
*refitted* parameters over the **original observed returns**, not over the synthetic
series. This keeps every replicate's forecast conditional on the real history actually
observed, so the bootstrap spread reflects parameter estimation uncertainty only — not
an unrelated random ending state of a synthetic path.

### 2.5 Monte Carlo coverage experiment

For R independent simulated paths (each a different "alternate history" under the same
true parameters), the full pipeline (simulate → fit → true VaR/ES → both bootstrap CIs)
is run, and it is recorded whether each CI contains its path's own true VaR/ES.
**Coverage** = the fraction of paths where it does. A correctly calibrated 95% interval
should achieve ≈95% coverage; this experiment tests whether it actually does, at
different sample sizes N.

### 2.6 Validity checks performed

- **Scaling test:** confirmed standardized innovations have unit variance over 10⁷ draws.
- **Big-simulation test:** closed-form true VaR/ES formula verified against brute-force
  empirical quantiles/tail means from 10⁷ draws (agreement to ~0.2%).
- **Conditional-forecast helper** verified against the fitted model's own point forecast.
- **Residual diagnostics (Ljung–Box):** standardized residuals show no significant
  remaining autocorrelation (~5% false-rejection rate, as expected under a correctly
  specified model). Squared standardized residuals show a mild excess false-rejection
  rate at small N (9% at N=500) that shrinks with N (7% at N=2500, 6% at N=5000),
  consistent with slow MLE convergence (see Section 5). The N=5000 point uses a
  separate, local fitting helper inside `diagnostics.py` (a different `arch` rescaling
  factor than `estimate.py`'s `fit_garch_t`, which is tuned for N ∈ {500, 1000, 2500});
  only `std_resid` is needed for this check, which is scale-invariant, so no
  back-scaling of `omega`/`sigma2_next` is required there.

## 3. Experiment configuration

| Parameter | Value |
|---|---|
| α, β, ν (true) | 0.08, 0.90, 6.0 |
| ω (derived) | ≈0.02 |
| Sample sizes N | 500, 1000, 2500 |
| VaR / ES levels | 99% / 97.5% |
| CI level | 95% |
| Burn-in | 500 |

Two full Monte Carlo runs were performed:

| Run | R (paths) | B (bootstrap replicates) |
|---|---|---|
| Run 1 | 300 | 399 |
| Run 2 (final) | 500 | 499 |

## 4. Results

### 4.1 Observed coverage (Run 2, R=500, B=499 — primary result)

| N | Parametric VaR | Parametric ES | Residual VaR | Residual ES |
|---|---|---|---|---|
| 500 | 0.894 | 0.890 | 0.896 | 0.892 |
| 1000 | 0.936 | 0.934 | 0.936 | 0.938 |
| 2500 | 0.930 | 0.930 | 0.924 | 0.928 |

Nominal target: 0.95 for all cells.

### 4.2 Statistical significance (two-sided binomial test against p=0.95)

| N | Parametric VaR | Parametric ES | Residual VaR | Residual ES |
|---|---|---|---|---|
| 500 | p<0.0001 ** | p<0.0001 ** | p<0.0001 ** | p<0.0001 ** |
| 1000 | p=0.150 | p=0.101 | p=0.150 | p=0.217 |
| 2500 | p=0.050 (borderline) | p=0.050 (borderline) | p=0.013 ** | p=0.031 ** |

(** = significant at 5%; Run 1 at R=300 showed the same qualitative pattern at N=500 and
N=1000 — all N=500 cells significant, all N=1000 cells non-significant — but found no
significant deviation at N=2500, where the larger R=500 run had more power to detect it.)

## 5. Conclusions

1. **At N=500, coverage is significantly and substantially below the nominal 95% level**
   (≈89–90% observed, p<0.0001 in every cell, both methods, both risk measures). Bootstrap
   confidence intervals built from a GARCH-t model fitted on 500 observations
   systematically understate the true estimation uncertainty in one-day VaR/ES.

2. **Coverage improves sharply by N=1000**, where no cell shows a statistically
   significant deviation from 95% (observed coverage ≈93–94%).

3. **At N=2500, a residual gap re-emerges**: coverage sits at ≈92–93%, which is
   statistically significant for the residual bootstrap and borderline (p≈0.05) for the
   parametric bootstrap. Coverage does **not** improve monotonically from N=1000 to
   N=2500 in this experiment — it is comparable or very slightly worse. This is not
   conclusively resolved: it may reflect (a) a genuine, slowly-vanishing residual
   under-coverage that only becomes statistically detectable with the larger R=500
   sample, (b) a flattening of the convergence curve rather than smooth monotonic
   improvement, or (c) sampling noise in this particular run. Distinguishing between
   these would require an independent replication at N=2500 with a different seed.

4. **No consistent ranking between the two bootstrap methods.** Differences between
   parametric and residual coverage are small (≤2 percentage points) and do not point in
   a consistent direction across N. This is expected: the data-generating process here
   genuinely is Student-*t*, so the parametric bootstrap's distributional assumption is
   exactly correct, removing the main theoretical advantage the residual (distribution-free)
   method would otherwise have.

5. **Mechanism.** The root cause of the small-N under-coverage is slow convergence of
   GARCH-t maximum likelihood estimation under ν=6. With these parameters, the condition
   for a finite fourth moment, (α+β)² + α²(κ_z−1) < 1, evaluates to ≈0.992 — very close
   to its boundary of 1. This implies very slow convergence of variance-based estimators
   (GARCH MLE is inherently one), which is independently confirmed by the Ljung–Box
   diagnostics on squared standardized residuals (Section 2.6), showing the same
   shrinking-with-N pattern. ν=6 was deliberately chosen from the typical empirical
   range (5–10) toward the fat-tailed end, specifically so this convergence effect would
   be visible within feasible sample sizes rather than hidden in noise.

## 6. Known limitations and planned extensions

- **Parameters are not yet calibrated to real data.** α, β, ν were chosen to resemble
  typical daily-equity GARCH fits, not estimated from WIG20/S&P 500. Planned: fit
  GARCH(1,1)-t to real index returns and recalibrate the simulation, or report results
  across a small robustness grid (e.g. ν ∈ {5, 8, 12}) to show the main conclusions are
  not an artifact of one specific, deliberately extreme parameter choice.
- **The N=2500 result is not fully resolved** (Conclusion 3) and would benefit from an
  independent-seed replication.
- **This part is purely simulation-based.** Part 2 (not yet started) applies the same
  pipeline — fit, bootstrap, interval construction — to real WIG20/S&P 500 data, where
  the true VaR/ES is of course unobservable, so evaluation will instead follow the
  backtesting approach used by Pascual et al. (comparing intervals against realized
  volatility / out-of-sample outcomes) rather than exact coverage.

## 7. References

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
  *Journal of Econometrics*, 31, 307–327.
- Pascual, L., Romo, J., & Ruiz, E. (2006). Bootstrap prediction for returns and
  volatilities in GARCH models. *Computational Statistics & Data Analysis*, 50(9),
  2293–2312.

## 8. Project structure

```
config.py       — Config dataclass: model/experiment parameters, validation, presets
simulate.py     — simulate_garch_t(): GARCH-t path generation; true_var_es() / 
                  var_es_from_params(): closed-form VaR/ES
estimate.py     — fit_garch_t(): MLE fit via `arch`, standardized residuals,
                  convergence flag; conditional_sigma2_next(): conditional forecast helper
bootstrap.py    — parametric_bootstrap(), residual_bootstrap()
coverage.py     — run_single_path(), coverage_metrics(): Monte Carlo coverage experiment
run.py          — full experiment runner, incremental results saved to JSON
diagnostics.py  — Ljung–Box diagnostics on standardized residuals; includes its own
                  local fit_for_diagnostics() helper (separate rescaling from
                  estimate.py's fit_garch_t) for the N=5000 robustness check
plot_coverage.py— coverage vs. N visualization
```

## 9. Reproducing the results

```python
from config import Config
from dataclasses import replace
from run import main  # or run run.py directly

cfg = Config()
run_cfg = replace(cfg, B=499)
# see run.py for the full experiment loop; results saved incrementally
# to coverage_results.json after each sample size N
```

All randomness is seeded (`np.random.SeedSequence`, spawned independently per path and
per bootstrap replicate), so results are fully reproducible given the same seed.