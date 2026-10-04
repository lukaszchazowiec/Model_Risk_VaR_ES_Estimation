import json
import time
import warnings
from dataclasses import replace

from config import Config
from coverage import coverage_metrics

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cfg = Config()
    run_cfg = replace(cfg, B=399)
    R = 300
    seed = 42

    all_results = {}

    print(f"=== Starting Monte Carlo Experiment (R={R}, B={run_cfg.B}) ===", flush=True)

    for N in cfg.sample_sizes:
        print(f"\n[+] Running for N={N}...", flush=True)
        start = time.time()

        metrics = coverage_metrics(N, run_cfg, seed=seed, R=R)

        elapsed = time.time() - start
        print(f"  -> Finished N={N} in {elapsed:.2f}s", flush=True)

        all_results[N] = metrics

        output = {
            "config": run_cfg.to_dict(),
            "R": R,
            "seed": seed,
            "results": all_results,
        }
        with open("coverage_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"  -> Saved progress for N={N} to coverage_results.json", flush=True)
        print(f"  -> Summary: {metrics}", flush=True)