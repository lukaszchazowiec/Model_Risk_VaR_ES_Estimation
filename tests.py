import json
import sys
from scipy.stats import binomtest

with open(sys.argv[1] if len(sys.argv) > 1 else "coverage_results.json") as f:
    data = json.load(f)

R = data["R"]
nominal = 0.95

print(f"R={R}, nominal coverage={nominal}\n")

for N_str, result in data["results"].items():
    summary = result["summary"] if "summary" in result else result
    n = summary.get("n_valid", R)
    print(f"--- N={N_str} (n_valid={n}) ---")
    for key, coverage in summary.items():
        if not key.endswith("coverage"):
            continue
        hits = round(coverage * n)
        test = binomtest(hits, n, p=nominal, alternative="two-sided")
        flag = " <-- significant" if test.pvalue < 0.05 else ""
        print(f"  {key}: {coverage:.4f} ({hits}/{n}), p-value={test.pvalue:.4f}{flag}")
    print()