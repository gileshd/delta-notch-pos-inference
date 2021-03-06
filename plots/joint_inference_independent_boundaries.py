import os
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from plotting_tools import plot_joint_with_marginals

plt.style.use("thesis")

OUT_FILE = "SDE_figures/joint_inference_independent_boundaries.pdf"

lims = (-5,5)

p = 0.9
τ = st.norm.ppf(p)
μ1, μ2 = -τ, τ
z1, z2 = [0,1], [1,0]
μ = np.array([μ1,μ2])
cov = np.diag([1,1])

ax_joint, ax_marg1, ax_marg2 = plot_joint_with_marginals(μ,cov,lims);

ax_joint.vlines((μ1+μ2)/2,*lims,ls="--",color="grey")
ax_joint.hlines((μ1+μ2)/2,*lims,ls="--",color="grey")

y_min, y_max = ax_marg1.get_ylim()
ax_marg1.vlines((μ1+μ2)/2,y_min,y_max,ls="--",color="grey")
ax_marg2.hlines((μ1+μ2)/2,y_min,y_max,ls="--",color="grey");

plt.savefig(OUT_FILE,bbox_inches='tight')
