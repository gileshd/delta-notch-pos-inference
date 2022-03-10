import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_optimality.pdf"


### LOAD DATA ###
RESULTS_DIR = "../../ab_performance_results/param_points_100"
N = 20
P_array = np.zeros((100,100,N))

for i in range(N):
    P_array[...,i] = np.loadtxt(f"{RESULTS_DIR}/P_np100_10e3_{i}.txt")

P = P_array.mean(-1)

### MAKE PLOT ###
β = 15.
O = np.exp(P*β)

N_param_samples = 100
a_array = np.logspace(-5,1, N_param_samples)
b_array = np.logspace(-1,5, N_param_samples)

A, B = np.meshgrid(a_array,b_array)

fig, ax = plt.subplots(figsize=(8,8))

ax.pcolormesh(A,B,O,shading="gouraud");
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("a",fontsize=20)
ax.set_ylabel("b",fontsize=20);

plt.savefig(OUT_FILE,bbox_inches="tight")
