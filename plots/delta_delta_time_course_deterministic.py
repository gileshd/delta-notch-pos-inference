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
from sde_systems.delta_notch_tools import plot_delta_delta_time_course

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_delta_time_course_deterministic.pdf"

t0, t1 = 0, 6
nsteps = 1000
ts = jnp.linspace(t0, t1, nsteps)

noise_scale=0.0
args = (0.01,100,4,4,noise_scale)

μ1, μ2 = 0.4,0.6
y0 = np.array([μ1,μ2,0.5,0.5])

rng = random.PRNGKey(0)

y = DN_sdeint_ito(y0,ts,rng,args,dt=0.01)

plot_delta_delta_time_course(y)

plt.xlim(0,1)
plt.ylim(0,1)

plt.savefig(OUT_FILE,bbox_inches="tight")
