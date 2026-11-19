import numpy as np
from scipy.stats import skew

from config import Config
from simulate import skewt_rvs, skewt_true_var_es, true_var_es

if __name__ == "__main__":
    cfg = Config()
    lam = -0.4
    sigma2_next = 1.0
    sigma_next = np.sqrt(sigma2_next)

    # 1) innowacje: srednia ~0, wariancja ~1, skosnosc ujemna dla lam < 0
    rng = np.random.default_rng(0)
    z = skewt_rvs(10_000_000, cfg.nu, lam, rng)
    print(f"mean={z.mean():.4f} (oczekiwane ~0), var={z.var():.4f} (oczekiwane ~1), "
          f"skosnosc={skew(z):.3f} (oczekiwana ujemna)")

    # 2) wzor zamkniety kontra brute force na 10 mln losowan
    r = sigma_next * z
    a_var = 1 - cfg.var_level
    a_es = 1 - cfg.es_level
    empirical_var = -np.percentile(r, a_var * 100)
    threshold = np.percentile(r, a_es * 100)
    empirical_es = -r[r <= threshold].mean()

    closed = skewt_true_var_es(sigma2_next, cfg.nu, lam, cfg)
    for name, emp, cl in (("VaR", empirical_var, closed["var"]), ("ES", empirical_es, closed["es"])):
        rel = (emp - cl) / cl
        print(f"{name} - empirical: {emp:.4f} closed-form: {cl:.4f} roznica wzgledna: {rel:+.3%}")
        assert abs(rel) < 0.01, f"{name}: roznica wieksza niz 1%"

    # 3) dla lam=0 wzor skew-t ma dawac to samo co zweryfikowany wzor symetrycznego t
    sym = true_var_es(sigma2_next, cfg)
    skew0 = skewt_true_var_es(sigma2_next, cfg.nu, 0.0, cfg)
    print(f"lam=0: VaR {skew0['var']:.4f} vs symetryczne {sym['var']:.4f}, "
          f"ES {skew0['es']:.4f} vs {sym['es']:.4f}")
    assert abs(skew0["var"] - sym["var"]) < 1e-3 and abs(skew0["es"] - sym["es"]) < 1e-3

    print("OK")