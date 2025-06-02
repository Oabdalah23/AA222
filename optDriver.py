#
# File: optDriver.py
# Name: Omar Abdallah
# Class: AA 222 / CS 361
# Project: MESC Wing-Box Optimization (camelCase style)
#

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from deap import base, creator, tools
import pandas as pd
import evaluate as eva  # our analytical evaluator

# ---------------- DOMAIN CONSTANTS ----------------
POP_SIZE = 40               # population size
N_GEN = 60                  # number of generations
CXPB = 0.9                  # crossover probability
MUTPB = 0.3                 # mutation probability
SEED = 22                   # RNG seed for reproducibility

# Decision-variable bounds (key order must match boundKeys)
BOUND_KEYS = ["tSkin", "nCells", "th1", "th2", "matId"]
BOUNDS = {
    "tSkin": (1.0, 5.0),    # mm (skin thickness)
    "nCells": (6, 32),      # integer (# of cells)
    "th1": (-90.0, 90.0),   # deg (first half-stack ply angle)
    "th2": (-90.0, 90.0),   # deg (second half-stack ply angle)
    "matId": (0.0, 1.0)     # 0=CFRP, 1=S2-glass (float→rounded)
}

# ---------------- DEAP SETUP ----------------
creator.create("FitnessTri", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessTri)
toolbox = base.Toolbox()

# Attribute generators
toolbox.register("attrTSkin", random.uniform, *BOUNDS["tSkin"])
toolbox.register("attrNCells", random.randint, *BOUNDS["nCells"])
toolbox.register("attrTh1", random.uniform, *BOUNDS["th1"])
toolbox.register("attrTh2", random.uniform, *BOUNDS["th2"])
toolbox.register("attrMatId", random.randint, 0, 1)

# Initialize one individual by cycling through each attribute generator once
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.attrTSkin,
     toolbox.attrNCells,
     toolbox.attrTh1,
     toolbox.attrTh2,
     toolbox.attrMatId),
    n=1
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------------- EVALUATION HELPERS ----------------
def clampIndividual(ind):
    """
    Clamp each gene to its bounds and force integer variables
    (nCells, matId). Returns the modified individual.
    """
    for i, key in enumerate(BOUND_KEYS):
        lo, hi = BOUNDS[key]
        ind[i] = max(lo, min(hi, ind[i]))
    ind[1] = int(round(ind[1]))  # nCells → integer
    ind[4] = int(round(ind[4]))  # matId → integer (0 or 1)
    return ind

def evaluateIndividual(ind):
    """
    DEAP-compatible wrapper for eva.evaluate(...).
    Input: [tSkin_mm, nCells, th1_deg, th2_deg, matId].
    Returns a tuple (mass, -EI, -kWh).
    """
    clampIndividual(ind)
    mass, negEI, negKWh = eva.evaluate(
        L=1.50,
        b=0.30,
        h=0.40,
        tSkin=ind[0] / 1000.0,   # convert mm → m
        N_cells=ind[1],
        plyT=0.125e-3,
        angs=[ind[2], ind[3]],
        matId=ind[4]
    )
    return mass, negEI, negKWh

# Register DEAP operators
toolbox.register("evaluate", evaluateIndividual)
toolbox.register("mate", tools.cxBlend, alpha=0.4)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    eta=20.0,
    low=[BOUNDS[k][0] for k in BOUND_KEYS],
    up=[BOUNDS[k][1] for k in BOUND_KEYS],
    indpb=0.25
)
toolbox.register("select", tools.selNSGA2)

# ---------------- NSGA-II MAIN LOOP ----------------
def runNsga(seed):
    """
    Run a three-objective NSGA-II (mass, –EI, –kWh) for POP_SIZE individuals
    over N_GEN generations. Returns the final Pareto population.
    """
    random.seed(seed)

    # 1) Initialize population and evaluate fitness
    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # 2) NSGA-II sorting (assign rank & crowding)
    pop = tools.selNSGA2(pop, k=len(pop))

    # 3) Evolution loop
    for gen in range(1, N_GEN + 1):
        # Every 10th generation, inject 10 new random individuals
        if gen % 10 == 0:
            extraPop = toolbox.population(n=10)
            for extInd in extraPop:
                extInd.fitness.values = toolbox.evaluate(extInd)
            pop.extend(extraPop)
            # Trim back to POP_SIZE via NSGA2
            pop = tools.selNSGA2(pop, k=POP_SIZE)

        # Parent selection via NSGA2
        parents = tools.selNSGA2(pop, k=len(pop))
        offspring = [toolbox.clone(ind) for ind in parents]

        # Crossover
        for i in range(0, len(offspring), 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i], offspring[i+1])
                clampIndividual(offspring[i])
                clampIndividual(offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                clampIndividual(ind)
                del ind.fitness.values

        # Evaluate invalid individuals
        invalidInds = [ind for ind in offspring if not ind.fitness.valid]
        for invalid in invalidInds:
            invalid.fitness.values = toolbox.evaluate(invalid)

        # Combine & select next generation
        pop = tools.selNSGA2(pop + offspring, k=POP_SIZE)

    return pop

# ---------------- ENHANCED VISUALIZATION FUNCTIONS ----------------
def createEnhancedParetoPlot(df, savePath="pareto3D_enhanced.png"):
    """
    Create an enhanced 3D scatter plot of the Pareto front with improved styling
    """
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    scatter = ax.scatter(
        df.mass, df.EI, df.kWh,
        c=df.kWh,
        cmap='plasma',
        s=60,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )

    ax.set_xlabel('Mass (kg)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Bending Stiffness EI (N·m²)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('Battery Energy (kWh)', fontsize=14, fontweight='bold', labelpad=10)

    ax.set_title(
        'Multi-Objective Optimization: Wing-Box Design\n'
        'Pareto Front (Mass vs. Stiffness vs. Energy)',
        fontsize=16, fontweight='bold', pad=20
    )

    ax.tick_params(axis='x', labelsize=11, colors='#2E2E2E')
    ax.tick_params(axis='y', labelsize=11, colors='#2E2E2E')
    ax.tick_params(axis='z', labelsize=11, colors='#2E2E2E')

    cbar = fig.colorbar(scatter, shrink=0.8, aspect=30, pad=0.1)
    cbar.set_label('Battery Energy (kWh)', fontsize=13, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('#E5E5E5')
    ax.yaxis.pane.set_edgecolor('#E5E5E5')
    ax.zaxis.pane.set_edgecolor('#E5E5E5')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    statsText = (
        f"Optimization Results:\n"
        f"• Solutions: {len(df)} designs\n"
        f"• Mass range: {df.mass.min():.1f} – {df.mass.max():.1f} kg\n"
        f"• EI range: {df.EI.min():.2f} – {df.EI.max():.2f} N·m²\n"
        f"• Energy range: {df.kWh.min():.3f} – {df.kWh.max():.3f} kWh"
    )
    ax.text2D(
        0.02, 0.98, statsText, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, ax

def create2DProjections(df, savePath="pareto_projections.png"):
    """
    Create 2D projections of the 3D Pareto front for detailed analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('', fontsize=16, fontweight='bold', y=0.95)

    # Mass vs EI
    scatter1 = axes[0,0].scatter(df.mass, df.EI, c=df.kWh, cmap='plasma',
                                 s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0,0].set_xlabel('Mass (kg)', fontweight='bold')
    axes[0,0].set_ylabel('Bending Stiffness EI (N·m²)', fontweight='bold')
    axes[0,0].set_title('Mass vs. Stiffness', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    fig.colorbar(scatter1, ax=axes[0,0], label='Energy (kWh)')

    # Mass vs kWh
    scatter2 = axes[0,1].scatter(df.mass, df.kWh, c=df.EI, cmap='viridis',
                                 s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0,1].set_xlabel('Mass (kg)', fontweight='bold')
    axes[0,1].set_ylabel('Battery Energy (kWh)', fontweight='bold')
    axes[0,1].set_title('Mass vs. Energy', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    fig.colorbar(scatter2, ax=axes[0,1], label='EI (N·m²)')

    # EI vs kWh
    scatter3 = axes[1,0].scatter(df.EI, df.kWh, c=df.mass, cmap='coolwarm',
                                 s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[1,0].set_xlabel('Bending Stiffness EI (N·m²)', fontweight='bold')
    axes[1,0].set_ylabel('Battery Energy (kWh)', fontweight='bold')
    axes[1,0].set_title('Stiffness vs. Energy', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    fig.colorbar(scatter3, ax=axes[1,0], label='Mass (kg)')

    # Design variables correlation
    scatter4 = axes[1,1].scatter(df.tSkin, df.nCells, c=df.mass, cmap='magma',
                                 s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[1,1].set_xlabel('Skin Thickness (mm)', fontweight='bold')
    axes[1,1].set_ylabel('Number of Cells', fontweight='bold')
    axes[1,1].set_title('Design Variables', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    fig.colorbar(scatter4, ax=axes[1,1], label='Mass (kg)')

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, axes

def createConvergencePlot(df, savePath="optimization_summary.png"):
    """
    Create a summary plot showing key optimization insights
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('', fontsize=18, fontweight='bold', y=0.95)

    # Material distribution
    matCounts = df.matId.value_counts()
    matLabels = ['CFRP', 'S2-Glass']
    axes[0,0].pie(matCounts.values,
                  labels=[matLabels[int(i)] for i in matCounts.index],
                  autopct='%1.1f%%', startangle=90,
                  colors=['#FF6B6B', '#4ECDC4'])
    axes[0,0].set_title('Material Distribution', fontweight='bold')

    # Skin thickness distribution
    axes[0,1].hist(df.tSkin, bins=15, alpha=0.7,
                   color='skyblue', edgecolor='black')
    axes[0,1].set_xlabel('Skin Thickness (mm)', fontweight='bold')
    axes[0,1].set_ylabel('Frequency', fontweight='bold')
    axes[0,1].set_title('Skin Thickness Distribution', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)

    # Cell count distribution
    axes[0,2].hist(df.nCells, bins=range(6, 34), alpha=0.7,
                   color='lightcoral', edgecolor='black')
    axes[0,2].set_xlabel('Number of Cells', fontweight='bold')
    axes[0,2].set_ylabel('Frequency', fontweight='bold')
    axes[0,2].set_title('Cell Count Distribution', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)

    # Objective correlations
    axes[1,0].scatter(df.mass, df.EI, alpha=0.6, c='purple', s=40)
    axes[1,0].set_xlabel('Mass (kg)', fontweight='bold')
    axes[1,0].set_ylabel('Bending Stiffness EI (N·m²)', fontweight='bold')
    axes[1,0].set_title('Mass-Stiffness Trade-off', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].scatter(df.mass, df.kWh, alpha=0.6, c='orange', s=40)
    axes[1,1].set_xlabel('Mass (kg)', fontweight='bold')
    axes[1,1].set_ylabel('Battery Energy (kWh)', fontweight='bold')
    axes[1,1].set_title('Mass-Energy Trade-off', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)

    axes[1,2].scatter(df.EI, df.kWh, alpha=0.6, c='green', s=40)
    axes[1,2].set_xlabel('Bending Stiffness EI (N·m²)', fontweight='bold')
    axes[1,2].set_ylabel('Battery Energy (kWh)', fontweight='bold')
    axes[1,2].set_title('Stiffness-Energy Trade-off', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savePath, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, axes

# ---------------- MAIN ENTRY POINT ----------------
if __name__ == "__main__":
    finalPopulation = runNsga(SEED)

    df = pd.DataFrame(
        [(*ind, *ind.fitness.values) for ind in finalPopulation],
        columns=[*BOUND_KEYS, "mass", "-EI", "-kWh"]
    )
    df["EI"] = -df["-EI"]
    df["kWh"] = -df["-kWh"]
    df.drop(columns=["-EI", "-kWh"]).to_csv("pareto3D.csv", index=False)

    createEnhancedParetoPlot(df)
    create2DProjections(df)
    createConvergencePlot(df)