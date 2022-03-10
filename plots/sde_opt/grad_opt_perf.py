import os
import sys
import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_grad_opt_perf.pdf"


RESULTS_DIR = "../../ab_grad_results/log_opt/lr01-1/"
param_array = np.loadtxt(f"{RESULTS_DIR}/param_log_opt2.txt") # [loga,logb]
perf_array = np.loadtxt(f"{RESULTS_DIR}/perf_log_opt2.txt")

fig, ax = plt.subplots()
itrs = np.arange(len(perf_array))

ax.plot(itrs,perf_array);

ax.set_xlabel("Gradient Update Step");
ax.set_ylabel("Performance");

plt.savefig(OUT_FILE,bbox_inches="tight")
