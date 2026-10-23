import argparse
import json
import time
import warnings
from dataclasses import replace

from config import Config
from coverage import coverage_metrics

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=int)
    p.add_argument("sizes", type=int, nargs="+")
    p.add_argument("--lam", type=float, default=None,
                   help="skosnosc innowacji skosne t; bez tej flagi zwykle t")
    args = p.parse_args()

    run_cfg = replace(Config(), B=499)
    R = 500
    dgp = "t" if args.lam is None else "skewt"
    lam = 0.0 if args.lam is None else args.lam
    tag = "" if args.lam is None else f"_skewt{args.lam}"

    print(f"=== seed={args.seed} R={R} B={run_cfg.B} dgp={dgp} lam={lam} N={args.sizes} ===", flush=True)

    for N in args.sizes:
        print(f"\n[+] N={N}...", flush=True)
        start = time.time()

        metrics = coverage_metrics(N, run_cfg, seed=args.seed, R=R, dgp=dgp, lam=lam)

        print(f"  -> N={N} done in {time.time() - start:.2f}s", flush=True)

        output = {"config": run_cfg.to_dict(), "R": R, "seed": args.seed,
                  "dgp": dgp, "lam": lam, "results": {N: metrics}}
        filename = f"coverage_results_N{N}_seed{args.seed}{tag}.json"
        with open(filename, "w") as f:
            json.dump(output, f)
        print(f"  -> Saved to {filename}", flush=True)