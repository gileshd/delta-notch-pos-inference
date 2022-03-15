import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_grad_opt_perf.pdf"


### LOAD PERF RESULTS ###
OPT_RESULTS_DIR = "../../ab_grad_results/log_opt/thresh_perf/"
OPT_FILE = "log_thresh_opt_ab0"
opt_results_files_suffixes = ["4_1","4_4","0_0-5", "0_4-8"]

perfs = np.zeros((200,len(opt_results_files_suffixes)))
for n,sfx in enumerate(opt_results_files_suffixes):
    perfs[...,n] = np.loadtxt(f"{OPT_RESULTS_DIR}/perf_{OPT_FILE}_{sfx}.txt")[:200]

### PLOT PERF RESULTS ###
fig, ax = plt.subplots()
itrs = np.arange(len(perfs))

for traj in perfs.T:
    ax.plot(itrs,traj,"C0",alpha=0.5);

ax.set_xlabel("Gradient Update Step");
ax.set_ylabel("Performance");

plt.savefig(OUT_FILE,bbox_inches="tight")
