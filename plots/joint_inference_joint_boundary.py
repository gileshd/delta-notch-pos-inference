import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from plotting_tools import plot_joint_with_marginals

plt.style.use("thesis")

OUT_FILE = "figures/joint_inference_joint_boundaries.pdf"

lims = (-5,5)

μ1, μ2 = -1.4, 1.4
z1, z2 = [0,1], [1,0]
μ = np.array([μ1,μ2])
cov = np.diag([1,1])

ax_joint, ax_marg1, ax_marg2 = plot_joint_with_marginals(μ,cov,lims);

ax_joint.plot(list(lims),list(lims),ls="--",color="grey");

plt.savefig(OUT_FILE,bbox_inches='tight')
