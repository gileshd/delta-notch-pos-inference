from jax import numpy as jnp
from jax import jit, vmap, random
from functools import partial

from sde_systems.delta_notch_tools import calculate_y0_std
from sde_systems.jax_delta_notch import DN_sdeint_ito, DN_sdeint_ito_fixed_grid


t0 = 0
t1 = 6
nsteps = 100
ts = jnp.linspace(t0, t1, nsteps)
ts_fg = jnp.linspace(t0, t1 * 2, nsteps * 2)

DN_sdeint_single = jit(lambda y0, key, args: DN_sdeint_ito(y0, ts, key, args, dt=0.001))

DN_sdeint = jit(
    vmap(lambda y0, key, args: DN_sdeint_ito(y0, ts, key, args, dt=0.001), (0, 0, None))
)

DN_sdeint_fg_single = jit(
    lambda y0, key, args: DN_sdeint_ito_fixed_grid(y0, ts_fg, key, args, dt=0.001)
)

DN_sdeint_fg = jit(
    vmap(
        lambda y0, key, args: DN_sdeint_ito_fixed_grid(y0, ts_fg, key, args, dt=0.001),
        (0, 0, None),
    )
)


@partial(jit, static_argnums=1)
def sample_y0_eq_jit(key, n_samples=1000):
    μ = jnp.array([0.5, 0.5])
    cov = jnp.array([[1, 0.5], [0.5, 1]]) * 0.03 ** 2
    y0_delta = random.multivariate_normal(key, μ, cov, shape=(n_samples,))
    y0_notch = jnp.ones((n_samples, 2)) * 0.5
    return jnp.hstack((y0_delta, y0_notch))


@jit
def sample_y0_eq_single(key):
    μ = jnp.array([0.5, 0.5])
    cov = jnp.array([[1, 0.5], [0.5, 1]]) * 0.03 ** 2
    y0_delta = random.multivariate_normal(key, μ, cov)
    y0_notch = jnp.ones(2) * 0.5
    return jnp.hstack((y0_delta, y0_notch))


μ = jnp.array([0.4, 0.6])
std = calculate_y0_std(*μ)
cov = jnp.diag(jnp.ones(2)) * std ** 2

@partial(jit, static_argnums=1)
def sample_y0_DN_jit(key, n_samples=1000):
    y0_delta = random.multivariate_normal(key, μ, cov, shape=(n_samples,))
    y0_notch = jnp.ones((n_samples, 2)) * 0.5
    return jnp.hstack((y0_delta, y0_notch))


@jit
def sample_args_perf(args, key, y0):
    n_samples = len(y0)
    keys = random.split(key, n_samples)

    ys = DN_sdeint(y0, keys, args)
    return mean_perf_y0y1(ys)


def args_perf_single_no_jit(args, key, y0):
    """ Calculate performance of `args` for sample `y0`."""
    ys = DN_sdeint_fg_single(y0, key, args)
    y0, y1 = ys[0, :2], ys[-1, :2]
    return compare_y0y1(y0, y1)


args_perf_single = jit(args_perf_single_no_jit)

noise_scale = 0.05
args = (0.01, 100, 4, 4, noise_scale)

ab_perf_single_no_jit = lambda a, b, key, y0: args_perf_single_no_jit(
    (a, b, *args[2:]), key, y0
)

ab_perf_single = jit(ab_perf_single_no_jit)


# Not jitting so that autoreload works.
def a_perf_single_no_jit(a, key, y0):
    """ Sample performance of parameter `a`."""
    args = (a, 100.0, 4.0, 4.0, 0.05)
    return args_perf_single(args, key, y0)


a_perf_single = jit(a_perf_single_no_jit)


@jit
def mean_perf_y0y1(ys):
    """
    Extract y0, y1 from ys and compare y0 to y1 for each.
    """
    y0 = ys[:, 0, :2]
    y1 = ys[:, -1, :2]
    perf = vmap(compare_y0y1)(y0, y1)
    return jnp.nanmean(perf)


@jit
def compare_y0y1(y0, y1):
    """
    A smooth version of `(dA0 < dB0) == (dA1, dB1)`.
    """
    dA0, dB0 = y0
    dA1, dB1 = y1
    return smooth_lt(dA1, dB1) * (dA0 < dB0) + smooth_gt(dA1, dB1) * (dA0 > dB0)


@jit
def smooth_lt(x, y):
    return sigmoid_tanh(y - x)


@jit
def smooth_gt(x, y):
    return sigmoid_tanh(x - y)


@jit
def sigmoid_tanh(x, scale=20):
    return 0.5 * (jnp.tanh(x * scale / 2) + 1)


@jit
def sigmoid_exp(x, scale=20):
    return 0.5 * (jnp.tanh(x * scale / 2) + 1)
