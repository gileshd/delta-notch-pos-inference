import os
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from plotting_tools import plot_gauss_curves

plt.style.use("thesis")

OUT_FILE = "SDE_figures/individual_inference.pdf"

p = 0.9
τ = st.norm.ppf(p)
μ1, μ2 = -τ, τ
μ = [μ1, μ2]
thresh = (μ1 + μ2) / 2

xlims = (-5,5)
g = np.linspace(*xlims,1000)

fig, ax = plt.subplots(1,1,figsize=(8,4))

ax = plot_gauss_curves(μ,xlims,ax=ax,colors=["C2","C5"]);

ax.vlines(thresh,0,0.5,ls='--',color="grey");

ax.set_xticks([μ1,μ2])
ax.set_xticklabels([r"$\mu_1$","$\mu_2$"])
ax.set_yticks([])

ax.set_xlim(*xlims);

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("$g$")
ax.set_ylabel("$p(g|z)$")

plt.legend();

plt.savefig(OUT_FILE,bbox_inches='tight')
