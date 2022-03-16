import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_optimality.pdf"

### LOAD DATA ###
P_RESULTS_DIR = "../../ab_performance_results/param_points_100/d1_thresh_0-5"
P_FILE = "P_thresh_np100_10e3"

N = 20
P_array = np.zeros((100,100,N))
for i in range(N):
    P_array[...,i] = np.loadtxt(f"{P_RESULTS_DIR}/{P_FILE}_{i}.txt")
P = P_array.mean(-1)

N_param_samples = 100
a_array = np.logspace(-5,1, N_param_samples)
b_array = np.logspace(-1,5, N_param_samples)

A, B = np.meshgrid(a_array,b_array)

fig, axs = plt.subplots(1,2,figsize=(8,4))

beta_array = [1.,10.]

#ax.pcolormesh(A,B,O,shading="gouraud");
for β,ax in zip(beta_array,axs):
    O = np.exp(P*β)
    ax.pcolormesh(A,B,O,shading="gouraud");
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("a",fontsize=15)
    ax.set_ylabel("b",fontsize=15);
    ax.text(0.07, 0.07, r"$\beta = {}$".format(β),color="w",transform=ax.transAxes)

plt.tight_layout()
plt.savefig(OUT_FILE,bbox_inches="tight")
