import numpy as np
from jax import numpy as jnp
from jax import jit, vmap, random
import jax

from sde_systems.jax_delta_notch import DN_sdeint_ito
from sde_systems.delta_notch_tools import (
    write_sde_results,
    calculate_y0_std,
    sample_y0_DN,
    drop_nans_DN
)

noise_scale = 0
args = (0.01, 100, 4, 4, noise_scale)


def run_delta_notch_sde(args, n_samples, z=0, dt=0.01, drop_nans=True):
    """ Simulate Delta-Notch system with `args`, `n_samples` times. """
    t0, t1 = 0, 6
    nsteps = 100
    ts = jnp.linspace(t0, t1, nsteps)

    μ1, μ2 = 0.4, 0.6
    std = calculate_y0_std(μ1, μ2)
    cov = np.diag([1, 1]) * std ** 2

    μ = np.array([μ1, μ2])
    z_idx = [[0, 1], [1, 0]][z]
    y0 = sample_y0_DN(*μ[z_idx], cov, n_samples)
    rng = random.PRNGKey(0)
    rngs = random.split(rng, n_samples)

    DN_sdeint = jit(vmap(lambda y0, rng: DN_sdeint_ito(y0, ts, rng, args, dt=dt)))

    ys = DN_sdeint(y0, rngs)
    if drop_nans:
        ys = drop_nans_DN(ys)

    return ys 


def run_delta_notch_sde_no_initial_noise(args, n_samples, z=0, dt=0.01, drop_nans=True):
    """ Simulate Delta-Notch system with `args`, `n_samples` times. """
    t0, t1 = 0, 6
    nsteps = 100
    ts = jnp.linspace(t0, t1, nsteps)

    μ1, μ2 = 0.4, 0.6
    cov = np.diag([1, 1]) * 0

    μ = np.array([μ1, μ2])
    z_idx = [[0, 1], [1, 0]][z]
    y0 = sample_y0_DN(*μ[z_idx], cov, n_samples)
    rng = random.PRNGKey(0)
    rngs = random.split(rng, n_samples)

    DN_sdeint = jit(vmap(lambda y0, rng: DN_sdeint_ito(y0, ts, rng, args, dt=dt)))

    ys = DN_sdeint(y0, rngs)
    if drop_nans:
        ys = drop_nans_DN(ys)

    return ys 

