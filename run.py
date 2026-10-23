import json
import sys
import time
import warnings
from dataclasses import replace

from config import Config
from coverage import coverage_metrics

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed = int(sys.argv[1])
    sizes = [int(x) for x in sys.argv[2:]]

    cfg = Config()
    run_cfg = replace(cfg, B=499)
    R = 500

    print(f"=== Monte Carlo (seed={seed}, R={R}, B={run_cfg.B}, N={sizes}) ===", flush=True)

    for N in sizes:
        print(f"\n[+] N={N}...", flush=True)
        start = time.time()

        metrics = coverage_metrics(N, run_cfg, seed=seed, R=R)

        print(f"  -> N={N} done in {time.time() - start:.2f}s", flush=True)
        print(f"  -> Summary: {metrics}", flush=True)

        output = {
            "config": run_cfg.to_dict(),
            "R": R,
            "seed": seed,
            "results": {N: metrics},
        }
        filename = f"coverage_results_N{N}_seed{seed}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  -> Saved to {filename}", flush=True)