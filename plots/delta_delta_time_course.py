import os
import sys
import jax; jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from jax import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.jax_delta_notch import DN_sdeint_ito

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_delta_time_course.pdf"

t0, t1 = 0, 6
nsteps = 1000
ts = jnp.linspace(t0, t1, nsteps)

args = (0.01,100,4,4)

μ1, μ2 = 0.4,0.6
y0 = np.array([μ1,μ2,0.5,0.5])

rng = random.PRNGKey(0)

y = DN_sdeint_ito(y0,ts,rng,args,dt=0.01)

d1, d2 = y[...,:2].T

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
points = np.array([d1, d2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
color = np.linspace(0,1,len(d1))


lc = LineCollection(segments, cmap='viridis')
lc.set_array(color)
lc.set_linewidth(2)

fig, ax = plt.subplots()
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax,fraction=0.05, shrink=0.9, pad=-0.2)

ax.axis('scaled');
lim_max = max(d1.max(),d2.max())
ax.set_xlim(0,lim_max);
ax.set_ylim(0,lim_max);

ax.set_xlabel(r"$\mathrm{Delta}_1$");
ax.set_ylabel(r"$\mathrm{Delta}_2$");

plt.savefig(OUT_FILE,bbox_inches="tight")
