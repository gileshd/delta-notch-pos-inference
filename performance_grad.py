from jax import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from jax import jit, grad, vmap, random
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from functools import partial

from sde_systems.jax_delta_notch_tools import a_perf_single_no_jit
from sde_tools.data_funcs import drop_nans

# Jitting here so that autoreload works.
# Need to run this again after change in other file.
a_perf_single = jit(lambda a, key: a_perf_single_no_jit(a,key))

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
print(estimate_grad_a(0.01,key1,n_samples=10000)) # 4.077
print(estimate_grad_a(0.01,key2,n_samples=10000)) # 0.015

key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(estimate_grad_a(0.1,key1,n_samples=10000)) # -0.433
print(estimate_grad_a(0.1,key2,n_samples=10000)) # -0.672

key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(estimate_grad_a(0.001,key1,n_samples=10000)) # 17.549
print(estimate_grad_a(0.001,key2,n_samples=10000)) # 2.141

key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(estimate_grad_a(0.3,key1,n_samples=10000)) # -0.123
print(estimate_grad_a(0.3,key2,n_samples=10000)) # -0.178

n_samples = 1000
key = random.PRNGKey(101)
keys = random.split(key,n_samples)
grad_array = vmap(grad(a_perf_single),(None,0))(0.01,keys)

grad_array_clean = drop_nans(grad_array)


key = random.PRNGKey(8)
a = 0.01
lr = 0.05
a_list = [a]

for _ in range(10):
    key, subkey = random.split(key)
    grad_a = estimate_grad_a(a,key,n_samples=20000)
    a += lr * grad_a
    print(a)
    a_list.append(a)

perf_0041 = sample_perf_a(0.041,key,n_samples=50000)
perf_0039 = sample_perf_a(0.039,key,n_samples=50000)
perf_0046 = sample_perf_a(0.046,key,n_samples=50000)
perf_0040 = sample_perf_a(0.040,key,n_samples=50000)
