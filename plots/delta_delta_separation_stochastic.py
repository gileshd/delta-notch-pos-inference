import os
import sys
import jax; jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from jax import random, jit, vmap
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sde_systems.jax_delta_notch import DN_sdeint_ito
from sde_systems.delta_notch_tools import plot_DN_state_space, sample_y0_DN

plt.style.use("thesis")

OUT_FILE = "SDE_figures/delta_delta_separation_stochastic.pdf"


DN_sdeint = jit(vmap(lambda y0, key, args: DN_sdeint_ito(y0,ts,key,args,dt=0.001),
                     (0,0,None)))

t0, t1 = 0, 6
nsteps = 1000
ts = jnp.linspace(t0, t1, nsteps)

noise_scale = 0.05
args = (0.01,100,4,4,noise_scale)

N_samples = 100
μ1, μ2 = 0.5,0.5
cov = np.array([[1,0.9],
                [0.9,1]]) * 0.02
np.random.seed(111)
y0 = sample_y0_DN(μ1,μ2,cov,N_samples)

key = random.PRNGKey(0)
keys = random.split(key,N_samples)

y = DN_sdeint(y0,keys,args)

dA0, dB0 = y[:,0,:2].T
dA1, dB1 = y[:,-1,:2].T
y_A = y[dA1 > dB1,...]
y_B = y[dA1 < dB1,...]

fig,ax = plt.subplots()
plot_DN_state_space(y_A,ax=ax,color="C1",alpha=0.05);
plot_DN_state_space(y_B,ax=ax,color="C0",alpha=0.05);
ax.plot([0,1],[0,1],color="black",ls="--",alpha=0.6);
ax.plot(*y_A[:,0,:2].T,'.',color="C1");
ax.plot(*y_B[:,0,:2].T,'.',color="C0");

ax.set_xlabel(r"$\mathrm{Delta_A}$")
ax.set_ylabel(r"$\mathrm{Delta_B}$");

plt.savefig(OUT_FILE,bbox_inches="tight")
