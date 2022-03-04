import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.delta_notch_tools import drop_nans_DN, load_sde_results
from plotting_tools import make_joint_and_marginal_axes, plot_joint_and_marginal_kde

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_delta_contour_t0t1.pdf"

genes = ["delta1","delta2","notch1","notch2"]
n_samples = 100000
ys = np.zeros((n_samples,100,4))
for n,g in enumerate(genes):
    RESULTS_FILE = f"../results/DN_trajectories_04-06_{g}.txt"
    ys[:,:,n] = np.loadtxt(RESULTS_FILE)
    
z_batch_size = int(n_samples/2)
ys_z0 = drop_nans_DN(ys[:z_batch_size,...])
ys_z1 = drop_nans_DN(ys[z_batch_size:,...])

ax_joint, ax_marg1, ax_marg2 = make_joint_and_marginal_axes()
axs = (ax_joint, ax_marg1, ax_marg2)

sample_size = 50000
y0_z0 = ys_z0[:sample_size,0,:2]
y1_z0 = ys_z0[:sample_size,-1,:2]
y0_z1 = ys_z1[:sample_size,0,:2]
y1_z1 = ys_z1[:sample_size,-1,:2]

plot_joint_and_marginal_kde(y0_z0, axs, t=0,z=0)
plot_joint_and_marginal_kde(y0_z1, axs, t=0,z=1)
plot_joint_and_marginal_kde(y1_z0, axs, t=1,z=0)
plot_joint_and_marginal_kde(y1_z1, axs, t=1,z=1)

ax_joint.axis('scaled');
ax_joint.set_xlabel(r"$\mathrm{Delta_A}$");
ax_joint.set_ylabel(r"$\mathrm{Delta_B}$");

handles,labels = ax_marg1.get_legend_handles_labels()
ax_joint.legend(handles,labels);

plt.savefig(OUT_FILE,bbox_inches="tight")
