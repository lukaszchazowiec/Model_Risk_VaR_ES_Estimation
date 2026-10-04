import json
from scipy.stats import binomtest

with open("coverage_results.json") as f:
    data = json.load(f)

R = data["R"]
nominal = 0.95

print(f"R={R}, nominal coverage={nominal}\n")

for N_str, result in data["results"].items():
    summary = result["summary"] if "summary" in result else result
    print(f"--- N={N_str} ---")
    for key, coverage in summary.items():
        if "coverage" not in key:
            continue
        hits = round(coverage * R)
        test = binomtest(hits, R, p=nominal, alternative="two-sided")
        flag = " <-- significant" if test.pvalue < 0.05 else ""
        print(f"  {key}: {coverage:.4f} ({hits}/{R}), p-value={test.pvalue:.4f}{flag}")
    print()