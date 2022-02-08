from jax import numpy as jnp
from jax import jit, vmap
from jaxsde.jaxsde import sdeint_ito

# noise_scale = 0.05
# args = (0.01, 100., 4., 4., noise_scale)


def f(y, t, args):
    prod = production(y, args)
    dec = decay(y)
    return prod - dec


def g(y, t, args):
    noise_scale = args[-1]
    prod = production(y, args)
    dec = decay(y)
    return jnp.sqrt(prod + dec) * noise_scale


def production(y, args):
    a, b, h, k, _ = args
    delta = delta_f(y[2:], b, h)
    notch = notch_f(jnp.flip(y[:2]), a, k)
    return jnp.hstack((delta, notch))


def delta_f(y, b, h):
    return 1 / (1 + b * y ** h)


def notch_f(y, a, k):
    return y ** k / (a + y ** k)


def decay(y):
    return y


def DN_sdeint_ito(y0, ts, rng, args, dt=0.001):
    return sdeint_ito(f, g, y0, ts, rng, args, dt)
