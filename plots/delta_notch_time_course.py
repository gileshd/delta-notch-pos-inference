import os
import sys
import jax; jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from jax import random
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.jax_delta_notch import DN_sdeint_ito

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_notch_time_course.pdf"

t0, t1 = 0, 6
nsteps = 1000
ts = jnp.linspace(t0, t1, nsteps)

args = (0.01,100,4,4)

μ1, μ2 = 0.4,0.6
y0 = np.array([μ1,μ2,0.5,0.5])

rng = random.PRNGKey(0)

y = DN_sdeint_ito(y0,ts,rng,args,dt=0.01)


gene_names = ["delta1","delta2","notch1","notch2"]
gene_label_dict = {
    "delta1":r"$\mathrm{Delta}_1$",
    "delta2":r"$\mathrm{Delta}_2$",
    "notch1":r"$\mathrm{Notch}_1$",
    "notch2":r"$\mathrm{Notch}_2$"
}

gene_color_dict = {
    "delta1":"#054d80",
    "delta2":"#6597b8",
    "notch1":"#ad0c00",
    "notch2":"#de7f78",
}

fig, ax = plt.subplots()
t = np.linspace(0,1,len(y))

for n, g in enumerate(y.T):
    gene = gene_names[n]
    ax.plot(t,g,
             color=gene_color_dict[gene],
             label=gene_label_dict[gene]);
    
ax.set_xlim(0,1)
ylims = ax.get_ylim()
ax.set_ylim(0,ylims[1])

ax.legend();
ax.set_xlabel("Time")
ax.set_ylabel("Expression");

plt.savefig(OUT_FILE,bbox_inches="tight")
