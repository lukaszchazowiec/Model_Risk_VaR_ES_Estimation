import numpy as np
import pandas as pd

from estimate import fit_garch_t

FILES = {
    "WIG20": "data/wig20_d.csv",
    "S&P 500": "data/sp500_2004_2025.csv",
}
WINDOW = 1000


def load_returns(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", utc=True)
    close_col = "Close" if "Close" in df.columns else "Zamkniecie"
    close = pd.to_numeric(df[close_col], errors="coerce")
    prices = pd.Series(close.values, index=dates).dropna().sort_index()
    return 100 * np.log(prices).diff().dropna()


def fit_and_print(label, r):
    f = fit_garch_t(r.to_numpy() - r.mean())
    print(f"{label}: alpha={f['alpha']:.3f} beta={f['beta']:.3f} "
          f"alpha+beta={f['alpha'] + f['beta']:.3f} nu={f['nu']:.2f} converged={f['converged']}")


if __name__ == "__main__":
    for name, path in FILES.items():
        r = load_returns(path)
        print(f"\n{name}: {len(r)} zwrotów, {r.index[0].date()} - {r.index[-1].date()}")
        fit_and_print("cała próba", r)
        for start in range(0, len(r) - WINDOW + 1, WINDOW):
            w = r.iloc[start:start + WINDOW]
            fit_and_print(f"{w.index[0].date()} - {w.index[-1].date()}", w)