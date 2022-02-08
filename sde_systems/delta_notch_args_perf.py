from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from jax import jit, vmap, grad, random
from jax import numpy as jnp
from sde_systems.jax_delta_notch import DN_sdeint_ito
from sde_systems.delta_notch_tools import calculate_y0_std

noise_scale = 0.05
a,b = 0.01, 100
h,k = 4,4
args = (a,b,h,k,noise_scale)

μ = jnp.array([0.4,0.6])
std = calculate_y0_std(*μ)
cov = jnp.diag(jnp.ones(2)) * std**2

n_samples = 10000
key = random.PRNGKey(0)

@jit
def sample_y0_DN_jit(key):
    y0_delta = random.multivariate_normal(key,μ,cov,shape=(n_samples,))
    y0_notch = jnp.ones((n_samples,2)) * 0.5
    return jnp.hstack((y0_delta,y0_notch))

t0, t1 = 0, 6
nsteps = 100
ts = jnp.linspace(t0, t1, nsteps)

DN_sdeint = jit(vmap(lambda y0, rng, args: DN_sdeint_ito(y0, ts, rng, args, dt=0.01),(0,0,None)))

@jit
def run_DN_sdeint_jit(args,key):
    keys = random.split(key, n_samples)
    
    y0 = sample_y0_DN_jit(key)
    return DN_sdeint(y0,keys,args)

@jit
def optimal_threshold_joint_perf_jit(dA,dB):
    """
    Calculate the joint performance of the optimal threshold to
     distinguish between the distribution of values in dA, dB.
     
    Deals with nans.
    """
    
    nan_mask = jnp.isnan(dA) | jnp.isnan(dB)
    
    sorted_idx = jnp.argsort(jnp.array([jnp.nanmedian(dA), jnp.nanmedian(dB)]))
    d1, d2 = jnp.array([dA, dB])[sorted_idx]
    
    τ = jnp.linspace(0, 1, 100)
    B = (d1[:, None] < τ[None, :]) & (d2[:, None] > τ[None, :])
    p_max = jnp.mean(B,axis=0,where=~nan_mask[:,None]).max()
    return p_max

@jit
def estimate_args_perf(args,key):
    ys = run_DN_sdeint_jit(args,key)
    
    dA, dB = ys[:,-1,:2].T
    return optimal_threshold_joint_perf_jit(dA,dB)

@jit
def estimate_a_perf(a,key):
    args = (a, 100.,4.,4.,0.05)
    ys = run_DN_sdeint_jit(args,key)
    
    dA, dB = ys[:,-1,:2].T
    return optimal_threshold_joint_perf_jit(dA,dB)


@jit
def sample_y0_eq_jit(key):
    μ = jnp.array([0.5,0.5])
    cov = jnp.diag(jnp.ones(2)) * std**2
    y0_delta = random.multivariate_normal(key,,cov,shape=(n_samples,))
    y0_notch = jnp.ones((n_samples,2)) * 0.5
    return jnp.hstack((y0_delta,y0_notch))
