import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from plotting_tools import plot_gauss_curves

plt.style.use("thesis")

OUT_FILE = "figures/individual_inference.pdf"

xlims = (-5,5)
g = np.linspace(*xlims,1000)
μ1, μ2 = -1.4, 1.4
μ = [μ1, μ2]
thresh = (μ1 + μ2) / 2

fig, ax = plt.subplots(1,1)

ax = plot_gauss_curves(μ,xlims,ax=ax);

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
