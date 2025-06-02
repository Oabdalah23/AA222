#
# File: postGp.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box GP Fitting
#

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# ---------------- CONSTANTS ----------------
archiveFilename = "pareto3D.csv"
numLhsPoints = 10_000
stdThresholdFrac = 0.05


def latinHypercube(nDim, nSamples, seed=42):
    """
    Generate nSamples×nDim Latin-Hypercube samples in [0, 1]^nDim.
    Each column is one randomly permuted stratum, then jittered.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((nSamples, nDim))
    for j in range(nDim):
        perm = rng.permutation(nSamples)
        jitter = rng.random(nSamples)
        X[:, j] = (perm + jitter) / nSamples
    return X


def fitGaussianProcesses(filename=archiveFilename):
    """
    1) Load the Pareto archive (CSV).
    2) Fit three GPRs on X = [tSkin, th1, th2, nCells, matId].
    3) Return (gpMass, gpEI, gpKWh, dfAll) without printing.
    """
    dfAll = pd.read_csv(filename)

    requiredCols = ["tSkin", "th1", "th2", "nCells", "matId", "mass", "EI", "kWh"]
    for col in requiredCols:
        if col not in dfAll.columns:
            raise KeyError(f"Column '{col}' not found in {filename}.")

    X = dfAll[["tSkin", "th1", "th2", "nCells", "matId"]].values
    y = dfAll[["mass", "EI", "kWh"]].values

    gpMass = GaussianProcessRegressor(kernel=1.0 * Matern(nu=2.5) + WhiteKernel(1e-5), normalize_y=True).fit(X, y[:, 0])
    gpEI   = GaussianProcessRegressor(kernel=1.0 * Matern(nu=2.5) + WhiteKernel(1e-5), normalize_y=True).fit(X, y[:, 1])
    gpKWh  = GaussianProcessRegressor(kernel=1.0 * Matern(nu=2.5) + WhiteKernel(1e-5), normalize_y=True).fit(X, y[:, 2])

    return gpMass, gpEI, gpKWh, dfAll


def computeSavedCallsFraction(gpMass, dfAll):
    """
    1) Generate numLhsPoints Latin-Hypercube in 5D.
    2) Scale to real bounds.
    3) Predict σ_mass at each point.
    4) Return fraction with σ_mass < (stdThresholdFrac × mean mass).
    """
    bounds = np.array([
        [1.0,   5.0],   # tSkin
        [-90.0, 90.0],  # th1
        [-90.0, 90.0],  # th2
        [6.0,   32.0],  # nCells
        [0.0,   1.0]    # matId
    ])

    unitLhs = latinHypercube(nDim=5, nSamples=numLhsPoints, seed=42)
    Xq = np.zeros_like(unitLhs)
    for j in range(5):
        lo, hi = bounds[j]
        Xq[:, j] = lo + unitLhs[:, j] * (hi - lo)

    _, stdMass = gpMass.predict(Xq, return_std=True)
    avgMass = dfAll["mass"].mean()
    threshold = stdThresholdFrac * avgMass
    pctSaved = np.mean(stdMass < threshold)
    return pctSaved


if __name__ == "__main__":
    gpMass, gpEI, gpKWh, dfData = fitGaussianProcesses()
    savedFraction = computeSavedCallsFraction(gpMass, dfData)