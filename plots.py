import json
import matplotlib.pyplot as plt

with open("coverage_results.json") as f:
    data = json.load(f)

Ns = sorted(int(n) for n in data["results"].keys())

series = {
    "parametric var coverage": [],
    "parametric es coverage": [],
    "residual var coverage": [],
    "residual es coverage": [],
}

for N in Ns:
    summary = data["results"][str(N)]
    for key in series:
        series[key].append(summary[key])

plt.figure(figsize=(12, 10))

for label, values in series.items():
    plt.plot(Ns, values, marker="o", label=label)

plt.axhline(0.95, color="black", linestyle="--", label="nominal 95%")

plt.xlabel("Sample size N")
plt.ylabel("Coverage")
plt.title("Bootstrap interval coverage vs sample size")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("coverage_plot.png", dpi=150)
plt.show()
