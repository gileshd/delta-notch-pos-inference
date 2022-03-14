import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_grad_opt_trajectory.pdf"

### LOAD RESULTS ###
P_RESULTS_DIR = "../../ab_performance_results/param_points_100"
N = 20
P_array = np.zeros((100,100,N))
for i in range(N):
    P_array[...,i] = np.loadtxt(f"{P_RESULTS_DIR}/P_np100_10e3_{i}.txt")
P = P_array.mean(-1)

OPT_RESULTS_DIR = "../../ab_grad_results/log_opt/lr1-1/"
opt_results_files_suffixes = ["0_5","4_4","0_0-5", "5_0-5"]

params = np.zeros((500,2,len(opt_results_files_suffixes)))
perfs = np.zeros((500,len(opt_results_files_suffixes)))
for n,sfx in enumerate(opt_results_files_suffixes):
    params[...,n] = np.loadtxt(f"{OPT_RESULTS_DIR}/param_log_opt_ab0_{sfx}.txt")[:500,:]
    perfs[...,n] = np.loadtxt(f"{OPT_RESULTS_DIR}/perf_log_opt_ab0_{sfx}.txt")[:500]

### PLOT RESULTS ###
N_param_samples = 100
a_array = np.logspace(-5,1, N_param_samples)
b_array = np.logspace(-1,5, N_param_samples)

A, B = np.meshgrid(a_array,b_array)

fig, ax = plt.subplots(figsize=(6,6))

ax.pcolormesh(A,B,P,shading="gouraud")

for n in range(params.shape[-1]):
    a, b = (10**params[...,n]).T
    ax.plot(a,b,'C5',alpha=0.7)
    ax.plot(a[0],b[0],'o',markersize=5,c='#fcf5fc')
    ax.scatter(a[-1],b[-1],marker='x',s=10,color='k',zorder=10)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("a",fontsize=20)
ax.set_ylabel("b",fontsize=20);

plt.savefig(OUT_FILE,bbox_inches="tight")
