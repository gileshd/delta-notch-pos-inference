import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.delta_notch_tools import drop_nans_DN, load_sde_results
from plotting_tools import make_joint_and_marginal_axes, plot_joint_and_marginal_kde

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_delta_contour_t0t1.pdf"


RESULTS_FILE = "../results/noise_experiments/delta_delta_t0t1_2/DD_t0t1_noise-0500_{}.txt"
ys_load = load_sde_results(RESULTS_FILE,genes=["deltaA","deltaB"])

ax_joint, ax_marg1, ax_marg2 = make_joint_and_marginal_axes()
axs = (ax_joint, ax_marg1, ax_marg2)

n_subsamples = 100000
y0_z1, y0_z2 = ys_load[:n_subsamples,0,:].T
y1_z1, y1_z2 = ys_load[:n_subsamples,1,:].T

y0_z12 = np.vstack((y0_z1, y0_z2)).T
y0_z21 = np.vstack((y0_z2, y0_z1)).T
y1_z12 = np.vstack((y1_z1, y1_z2)).T
y1_z21 = np.vstack((y1_z2, y1_z1)).T

plot_joint_and_marginal_kde(y0_z12, axs, t=0,z=0)
plot_joint_and_marginal_kde(y0_z21, axs, t=0,z=1)
plot_joint_and_marginal_kde(y1_z12, axs, t=1,z=0)
plot_joint_and_marginal_kde(y1_z21, axs, t=1,z=1)

ax_joint.axis('scaled');
ax_joint.set_xlabel(r"$\mathrm{Delta_A}$");
ax_joint.set_ylabel(r"$\mathrm{Delta_B}$");

handles,labels = ax_marg1.get_legend_handles_labels()
ax_joint.legend(handles,labels);

plt.savefig(OUT_FILE,bbox_inches="tight")
