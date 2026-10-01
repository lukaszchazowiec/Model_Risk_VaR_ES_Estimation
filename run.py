import json
import time
from dataclasses import replace

from config import Config
from coverage import coverage_metrics

if __name__ == "__main__":
    cfg = Config()
    run_cfg = replace(cfg, B=100)
    R = 40
    seed = 42

    all_results = {}

    for N in cfg.sample_sizes:
        print(f"Running for N={N} sample sizes...")
        start = time.time()

        metrics = coverage_metrics(N, run_cfg, seed=seed, R=R)

        elapsed = time.time() - start
        print(f"  done in {elapsed:.2f}s: {metrics['summary']}")

        all_results[N] = metrics

    output = {
        "config": run_cfg.to_dict(),
        "R": R,
        "seed": seed,
        "results": all_results,
    }

    with open("coverage_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved results to coverage_results.json")