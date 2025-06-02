#
# File: spearmanImportance.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box Spearman Importance
#

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ---------------- LOAD PARETO ARCHIVE ----------------
df = pd.read_csv("pareto3D.csv")

# ---------------- COMPUTE SPEARMAN RHO ----------------
varsToCheck = ["tSkin", "nCells", "th1", "th2", "matId"]
massSeries = df["mass"]

spearmanResults = {}
for var in varsToCheck:
    rho, _ = spearmanr(df[var], massSeries)
    spearmanResults[var] = rho

impDf = pd.DataFrame.from_dict(
    spearmanResults, orient="index", columns=["spearmanRho"]
).sort_values("spearmanRho", key=abs, ascending=False)

# ---------------- SAVE AND PLOT IMPORTANCE ----------------
impDf.to_csv("spearman_importance.csv")

impDf["spearmanRho"].plot(
    kind="bar",
    color="tab:blue",
    edgecolor="black",
    linewidth=0.5,
    figsize=(4, 3)
)
plt.ylabel("Spearman ρ with Mass")
plt.title("Variable Importance (Mass Objective)")
plt.tight_layout()
plt.savefig("importance.png", dpi=300)