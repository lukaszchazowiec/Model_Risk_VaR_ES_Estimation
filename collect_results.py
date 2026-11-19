import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

RAW_DIR = Path("results/raw")
OUT_FILE = Path("results/wyniki_symulacji.csv")
NOMINAL = 0.95
MIN_R = 100          # pliki z mniejszym R to testy bramkowe i sa pomijane

METHODS = ["parametric", "residual", "bayesian"]
MODES = ["theoretical", "empirical"]
WORLDS = {
    (0.08, 0.90, 6.0): "baza",
    (0.06, 0.93, 8.0): "jak WIG20",
    (0.13, 0.86, 6.0): "jak S&P 500",
}


def world_name(cfg, dgp):
    if dgp == "skewt":
        return "skew-t (baza)"
    return WORLDS.get((cfg["alpha"], cfg["beta"], cfg["nu"]), "inny")


def add_row(rows, meta, N, n_paths, method, mode, measure, coverage, width, rel_width, code):
    hits = round(coverage * n_paths)
    rows.append({
        **meta,
        "N": int(N), "n_paths": n_paths,
        "method": method, "quantile": mode, "measure": measure.upper(),
        "coverage": coverage,
        "p_value_vs_0.95": binomtest(hits, n_paths, NOMINAL).pvalue,
        "mean_width": width, "mean_rel_width": rel_width,
        "code_version": code,
    })


if __name__ == "__main__":
    rows = []

    for path in sorted(RAW_DIR.glob("coverage_results*.json")):
        with open(path) as f:
            data = json.load(f)

        if data["R"] < MIN_R:
            print(f"pomijam (test bramkowy): {path.name}")
            continue

        cfg = data["config"]
        dgp = data.get("dgp", "t")
        meta = {
            "file": path.name,
            "world": world_name(cfg, dgp),
            "dgp": dgp, "lambda": data.get("lam", 0.0),
            "alpha": cfg["alpha"], "beta": cfg["beta"], "nu": cfg["nu"],
            "omega": cfg["omega"], "burn_in": cfg["burn_in"],
            "seed": data["seed"], "R": data["R"], "B": cfg["B"],
        }
        print(f"OK: {path.name} -> {meta['world']}, seed={meta['seed']}, "
              f"R={meta['R']}, B={meta['B']}, N={list(data['results'])}")

        for N, res in data["results"].items():
            if "summary" in res:
                # nowy format: 3 metody x 2 tryby kwantyla, szerokosci, n_valid
                s = res["summary"]
                for method in METHODS:
                    for mode in MODES:
                        for measure in ("var", "es"):
                            p = f"{method} {mode} {measure}"
                            add_row(rows, meta, N, s["n_valid"], method, mode, measure,
                                    s[f"{p} coverage"], s[f"{p} mean width"],
                                    s[f"{p} mean rel width"], "nowy")
            else:
                # stary format: 2 metody, tylko kwantyl teoretyczny, bez szerokosci
                for method in ("parametric", "residual"):
                    for measure in ("var", "es"):
                        add_row(rows, meta, N, data["R"], method, "theoretical", measure,
                                res[f"{method} {measure} coverage"], None, None, "stary")

    if not rows:
        raise SystemExit(f"Nie znaleziono plikow coverage_results*.json w: {RAW_DIR.resolve()}")

    df = pd.DataFrame(rows).sort_values(
        ["world", "N", "method", "quantile", "measure", "seed"])
    df.to_csv(OUT_FILE, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"\nZapisano {len(df)} wierszy do {OUT_FILE}")