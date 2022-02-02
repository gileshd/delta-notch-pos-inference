import numpy as np
from jax import numpy as jnp
from jax import jit, vmap, random
import jax

from sde_systems.jax_delta_notch import DN_sdeint_ito
from sde_systems.delta_notch_tools import (
    write_sde_results,
    load_sde_results,
    calculate_y0_std,
    sample_y0_DN_mixed,
)

noise_scale = 0
args = (0.01, 100, 4, 4, noise_scale)

def run_delta_notch_sde(args, n_samples):
    """ Simulate Delta-Notch system with `args`, `n_samples` times. """
    t0, t1 = 0, 6
    nsteps = 100
    ts = jnp.linspace(t0, t1, nsteps)

    μ1, μ2 = 0.4,0.6
    std = calculate_y0_std(μ1,μ2)
    cov = np.diag([1,1]) * std**2
    z_batch_size = int(n_samples/2)

    y0 = sample_y0_DN_mixed(μ1, μ2, cov, z_batch_size)
    rng = random.PRNGKey(0)
    rngs = random.split(rng, n_samples)

    DN_sdeint = jit(vmap(lambda y0, rng: DN_sdeint_ito(y0, ts, rng, args, dt=0.01)))

    return DN_sdeint(y0,rngs)
