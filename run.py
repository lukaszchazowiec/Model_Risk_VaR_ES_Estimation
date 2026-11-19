import argparse
import json
import time
from dataclasses import replace

import os
os.makedirs("results/raw", exist_ok=True)

from config import Config
from coverage import coverage_metrics

import warnings
warnings.filterwarnings("ignore", message=".*poorly scaled.*")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("sizes", type=int, nargs="+")
    p.add_argument("--lam", type=float, default=None,
                   help="skosnosc innowacji skosne t; bez tej flagi zwykle t")
    p.add_argument("--R", type=int, default=500)
    p.add_argument("--B", type=int, default=499)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--burn_in", type=int, default=None)
    args = p.parse_args()

    # tylko podane flagi nadpisuja domyslna konfiguracje
    given = {"alpha": args.alpha, "beta": args.beta, "nu": args.nu, "burn_in": args.burn_in}
    overrides = {k: v for k, v in given.items() if v is not None}

    run_cfg = replace(Config(), B=args.B, **overrides)
    R = args.R
    dgp = "t" if args.lam is None else "skewt"
    lam = 0.0 if args.lam is None else args.lam
    tag = ("" if args.lam is None else f"_skewt{args.lam}") \
        + "".join(f"_{k}{v}" for k, v in overrides.items()) \
        + f"_R{R}_B{args.B}"

    print(f"=== seed={args.seed} R={R} B={run_cfg.B} dgp={dgp} lam={lam} "
          f"alpha={run_cfg.alpha} beta={run_cfg.beta} nu={run_cfg.nu} burn_in={run_cfg.burn_in} "
          f"N={args.sizes} ===", flush=True)

    for N in args.sizes:
        print(f"\n[+] N={N}...", flush=True)
        start = time.time()

        metrics = coverage_metrics(N, run_cfg, seed=args.seed, R=R, dgp=dgp, lam=lam)

        print(f"  -> N={N} done in {time.time() - start:.2f}s", flush=True)

        output = {"config": run_cfg.to_dict(), "R": R, "seed": args.seed,
                  "dgp": dgp, "lam": lam, "results": {N: metrics}}
        filename = f"results/raw/coverage_results_N{N}_seed{args.seed}{tag}.json"
        with open(filename, "w") as f:
            json.dump(output, f)
        print(f"  -> Saved to {filename}", flush=True)