import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_grad_opt_perf.pdf"


### LOAD PERF RESULTS ###
OPT_RESULTS_DIR = "../../ab_grad_results/log_opt/lr1-1/"
opt_results_files_suffixes = ["0_5","4_4","0_0-5", "5_0-5"]

perfs = np.zeros((500,len(opt_results_files_suffixes)))
for n,sfx in enumerate(opt_results_files_suffixes):
    perfs[...,n] = np.loadtxt(f"{OPT_RESULTS_DIR}/perf_log_opt_ab0_{sfx}.txt")[:500]

### PLOT PERF RESULTS ###
fig, ax = plt.subplots()
itrs = np.arange(len(perfs))

for traj in perfs.T:
    ax.plot(itrs,traj,"C0",alpha=0.5);

ax.set_xlabel("Gradient Update Step");
ax.set_ylabel("Performance");

plt.savefig(OUT_FILE,bbox_inches="tight")
