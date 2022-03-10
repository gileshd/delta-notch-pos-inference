import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_grad_opt_trajectory.pdf"


### LOAD PERF RESULTS ###
P_RESULTS_DIR = "../../ab_performance_results/param_points_100"
N = 20
P_array = np.zeros((100,100,N))
for i in range(N):
    P_array[...,i] = np.loadtxt(f"{P_RESULTS_DIR}/P_np100_10e3_{i}.txt")
P = P_array.mean(-1)

### LOAD PARAM RESULTS ###
OPT_RESULTS_DIR = "../../ab_grad_results/log_opt/lr01-1/"
param_array = np.loadtxt(f"{OPT_RESULTS_DIR}/param_log_opt2.txt")
perf_array = np.loadtxt(f"{OPT_RESULTS_DIR}/perf_log_opt2.txt")

N_param_samples = 100
a_array = np.logspace(-5,1, N_param_samples)
b_array = np.logspace(-1,5, N_param_samples)
A, B = np.meshgrid(a_array,b_array)

### PLOT ###
fig, ax = plt.subplots(figsize=(6,6))

ax.pcolormesh(A,B,P,shading="gouraud")

a, b = (10**param_array).T
ax.plot(a,b,'C5')

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("a",fontsize=20)
ax.set_ylabel("b",fontsize=20);

plt.savefig(OUT_FILE,bbox_inches="tight")
