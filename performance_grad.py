from jax import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

from jax import jit, grad, vmap, random
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from functools import partial

from sde_systems.jax_delta_notch_tools import a_perf_single_no_jit, sample_y0_eq_single, sample_y0_eq_jit
from sde_tools.data_funcs import drop_nans

# Jitting here so that autoreload works.
# Need to run this again after change in other file.
a_perf_single = jit(lambda a, key, y0: a_perf_single_no_jit(a,key, y0))

key = random.PRNGKey(111)

y0 = sample_y0_eq_single(key)

a_perf_single(0.01,key,y0)

@jit
def sample_perf_a(a, key, y0):
    n_samples = len(y0)
    keys = random.split(key,n_samples)
    perf_array = vmap(a_perf_single,(None,0,0))(a,keys,y0)
    return jnp.nanmean(perf_array)

y0 = sample_y0_eq_jit(key)
key1 = random.PRNGKey(101)
key2 = random.PRNGKey(102)
print(sample_perf_a(0.01,key1,y0))
print(sample_perf_a(0.01,key2,y0))

y0 = sample_y0_eq_single(key)
print(grad(a_perf_single)(0.01,key,y0))

@jit
def estimate_grad_a(a,key,y0):
    n_samples = len(y0)
    keys = random.split(key,n_samples)
    grad_array = vmap(grad(a_perf_single),(None,0,0))(a,keys,y0)
    return jnp.nanmean(grad_array)

key = random.PRNGKey(101)
y0 = sample_y0_eq_jit(key,n_samples=10000)
key1, key2 = random.split(key)
print(estimate_grad_a(0.01,key1,y0)) # 3.363
print(estimate_grad_a(0.01,key2,y0)) # 3.125

key = random.PRNGKey(101)
y0 = sample_y0_eq_jit(key,n_samples=10000)
key1, key2 = random.split(key)
print(estimate_grad_a(0.1,key1,y0)) # -0.227
print(estimate_grad_a(0.1,key2,y0)) # -0.165

key = random.PRNGKey(101)
y0 = sample_y0_eq_jit(key,n_samples=10000)
key1, key2 = random.split(key)
print(estimate_grad_a(0.001,key1,y0)) # 3.421
print(estimate_grad_a(0.001,key2,y0)) # 16.481

key = random.PRNGKey(101)
y0 = sample_y0_eq_jit(key,n_samples=10000)
key1, key2 = random.split(key)
print(estimate_grad_a(0.3,key1,y0)) # -0.182
print(estimate_grad_a(0.3,key2,y0)) # -0.260


key = random.PRNGKey(8)
a = 0.01
lr = 0.05
a_list = [a]

y0 = sample_y0_eq_jit(key,n_samples=20000)

for _ in range(10):
    key, subkey = random.split(key)
    grad_a = estimate_grad_a(a,key,y0)
    a += lr * grad_a
    print(a)
    a_list.append(a)

