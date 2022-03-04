import os
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.delta_notch_tools import load_sde_results, calculate_y0_std

plt.style.use("thesis")

OUT_FILE = "SDE_figures/marginal_delta_t0t1.pdf"

RESULTS_FILE = "../results/noise_experiments/delta_delta_t0t1_2/DD_t0t1_noise-0500_{}.txt"
ys_load = load_sde_results(RESULTS_FILE,genes=["deltaA","deltaB"])

x = np.linspace(0,1.3,1000)


n_subsamples = len(ys_load)
y0_z1, y0_z2 = ys_load[:n_subsamples,0,:].T
y1_z1, y1_z2 = ys_load[:n_subsamples,1,:].T

y1_z1_mask = y1_z1 < 0.5
y1_z2_mask = y1_z2 < 0.5

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

μ1, μ2 = 0.4, 0.6
σ = calculate_y0_std(μ1, μ2)
f0_z1 = st.norm(loc=μ1, scale=σ).pdf
f0_z2 = st.norm(loc=μ2, scale=σ).pdf

ax1.plot(x,f0_z1(x), color="C2", label="$z=1$, $t=0$");
ax1.plot(x,f0_z2(x), color="C5", label="$z=2$, $t=0$");

f1_z1_0 = lambda x: st.kde.gaussian_kde(y1_z1[y1_z1_mask],bw_method=1.5)(x) * sum(y1_z1_mask)/len(y1_z1)
f1_z1_1 = lambda x: st.kde.gaussian_kde(y1_z1[~y1_z1_mask])(x) * sum(~y1_z1_mask)/len(y1_z1)
ax2.plot(x,f1_z1_0(x), color="C2", label="$z=1$, $t=1$");
ax2.plot(x,f1_z1_1(x), color="C2");

f1_z2_0 = lambda x: st.kde.gaussian_kde(y1_z2[y1_z2_mask])(x) * sum(y1_z2_mask)/len(y1_z2)
f1_z2_1 = lambda x: st.kde.gaussian_kde(y1_z2[~y1_z2_mask])(x) * sum(~y1_z2_mask)/len(y1_z2)
ax2.plot(x,f1_z2_0(x), color="C5", label="$z=2$, $t=1$");
ax2.plot(x,f1_z2_1(x), color="C5");

ax1.set_xlim((0,1.0));
ax2.set_xlim((0,1.3));

for ax in (ax1,ax2):
    ylims = ax.get_ylim()
    ax.vlines(0.5,0,ylims[1],ls="--",color="grey")
    ax.set_ylim((0,ylims[1]));
    ax.set_yticks([])
    ax.set_xlabel("Delta, $d$")
    ax.set_ylabel("$p(d | z)$")
    ax.legend()

plt.savefig(OUT_FILE,bbox_inches="tight")
