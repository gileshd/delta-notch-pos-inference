import os
import sys
import numpy as np
from jax import config

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)
from jax import random
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], "../.."))

from sde_systems.jax_delta_notch_tools import sample_y0_DN_jit, DN_sdeint
from sde_systems.delta_notch_tools import drop_nans_DN, plot_DN_state_space

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/ab_param_DD_samples.pdf"

loga, logb = -1.4, 2.0
args1 = (10 ** loga, 10 ** logb, 4.0, 4.0, 0.05)

a, b = 0.1, 10e3
args2 = (a, b, 4.0, 4.0, 0.05)

n_samples = 20

key = random.PRNGKey(111)
keys = random.split(key, n_samples)
y0 = sample_y0_DN_jit(key, n_samples=n_samples)

ys_ab1 = DN_sdeint(y0,keys,args1)
ys_ab2 = DN_sdeint(y0,keys,args2)

# Make sure we are using the same initial samples for each.
nan_mask1 = np.isnan(ys_ab1).any((1, 2))
nan_mask2 = np.isnan(ys_ab2).any((1, 2))
nan_mask_comb = nan_mask1 | nan_mask2

ys_ab1, ys_ab2 = ys_ab1[~nan_mask_comb], ys_ab2[~nan_mask_comb]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

dA1, dB1 = ys_ab1[:, -1, :2].T
plot_DN_state_space(ys_ab1[dA1 < dB1], ax=ax1, alpha=0.6)
if (dA1 > dB1).any():
    plot_DN_state_space(ys_ab1[dA1 > dB1], ax=ax1, color="C5", alpha=0.6)

dA1, dB1 = ys_ab2[:, -1, :2].T
plot_DN_state_space(ys_ab2[dA1 < dB1], ax=ax2, alpha=0.6)
if (dA1 > dB1).any():
    plot_DN_state_space(ys_ab2[dA1 > dB1], ax=ax2, color="C5", alpha=0.6)

plt.tight_layout()

plt.savefig(OUT_FILE, bbox_inches="tight")
