#
# File: evaluate.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box
#

import numpy as np

# ---------------- MATERIAL CONSTANTS ----------------
LAMINA = {
    0: dict(name="CFRP", E1=135e9, E2=10e9, G12=5.2e9, nu12=0.30, rho=1570.0),
    1: dict(name="S2G", E1=50e9,  E2=15e9, G12=4.0e9, nu12=0.28, rho=2050.0)
}
MASS_PER_CELL = 0.048           # kg (18650 form-factor)
ENERGY_PER_CELL = 3.7 * 2.5 / 1000.0  # kWh (3.7 V · 2.5 Ah)

# --------- CLASSICAL LAMINATE THEORY HELPER -----------

def computeAD11(angles, plyT, E1, E2, G12, nu12):
    """
    Return (A11, D11) for a symmetric laminate.
    """
    nu21 = nu12 * E2 / E1
    Q11 = E1 / (1 - nu12 * nu21)
    Q22 = E2 / (1 - nu12 * nu21)
    Q12 = nu12 * E2 / (1 - nu12 * nu21)
    Q66 = G12

    A11 = 0.0
    D11 = 0.0
    zBot = -len(angles) * plyT / 2.0

    for theta in angles:
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        Qbar11 = (
            Q11 * c**4
            + 2 * (Q12 + 2 * Q66) * s**2 * c**2
            + Q22 * s**4
        )
        zTop = zBot + plyT
        A11 += Qbar11 * (zTop - zBot)
        D11 += Qbar11 * (zTop**3 - zBot**3) / 3.0
        zBot = zTop

    return A11, D11

# ------------------ MASTER EVALUATOR ------------------

def evaluate(L, b, h, tSkin, N_cells, plyT, angs, matId=0):
    """
    Returns (mass, -EI, -kWh) based on closed-form formulas.
    """
    data = LAMINA[int(matId)]
    E1 = data["E1"]
    E2 = data["E2"]
    G12 = data["G12"]
    nu12 = data["nu12"]
    rho = data["rho"]

    # Build full symmetric stack of angles
    fullAngles = angs + angs[::-1]
    A11, _ = computeAD11(fullAngles, plyT, E1, E2, G12, nu12)

    # Effective laminate modulus in bending direction
    tLam = len(fullAngles) * plyT
    Eeff = A11 / tLam

    # Geometric moment of inertia (two skins carrying bending)
    Igeom = 2 * (b * tSkin) * (h / 2.0)**2
    EI = Eeff * Igeom

    # Composite mass (skins only)
    area = 2.0 * L * (b + h)  # two skins wrap front + back
    mComp = rho * area * tSkin

    # Battery mass
    mBatt = N_cells * MASS_PER_CELL

    # Battery energy storage
    kWh = N_cells * ENERGY_PER_CELL

    totalMass = mComp + mBatt

    # Return objectives (mass, -EI, -kWh)
    return totalMass, -EI, -kWh

# ---------------- QUICK CLI TEST --------------------
if __name__ == "__main__":
    # Example usage:
    _ , _ = evaluate(
        1.5, 0.3, 0.4,
        tSkin=0.003,    # 3 mm
        N_cells=20,
        plyT=0.125e-3,
        angs=[0, 45],
        matId=0
    )
