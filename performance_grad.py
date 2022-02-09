from jax import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from jax import jit, grad, vmap, random
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from functools import partial

from sde_systems.jax_delta_notch_tools import a_perf_single
from sde_tools.data_funcs import drop_nans


key = random.PRNGKey(111)

a_perf_single(0.01,key)

@partial(jit, static_argnums=2)
def sample_perf_a(a, key, n_samples=1000):
    keys = random.split(key,n_samples)
    perf_array = vmap(a_perf_single,(None,0))(a,keys)
    return jnp.nanmean(perf_array)

key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(sample_perf_a(0.01,key1))
print(sample_perf_a(0.01,key2))


gp = grad(a_perf_single)(0.01,key)

@partial(jit, static_argnums=2)
def estimate_grad_a(a,key,n_samples=1000):
    keys = random.split(key,n_samples)
    grad_array = vmap(grad(a_perf_single),(None,0))(a,keys)
    return jnp.nanmean(grad_array)

key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(estimate_grad_a(0.01,key1))
print(estimate_grad_a(0.01,key2))

n_samples = 1000
key = random.PRNGKey(101)
keys = random.split(key,n_samples)
grad_array = vmap(grad(a_perf_single),(None,0))(0.01,keys)


