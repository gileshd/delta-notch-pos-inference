import numpy as np
import jax; jax.config.update('jax_platform_name', 'cpu') 
from sde_systems.delta_notch_tools import write_sde_results
from sde_systems.run_delta_notch import run_delta_notch_sde

RESULT_DIR = "results/noise_experiments"

noise_range = np.linspace(0,0.25,11)
n_samples = 100000

for noise_scale in noise_range:
    np.random.seed(123)
    args = (0.01, 100, 4, 4, noise_scale)

    print(f"Running noise={noise_scale:.4f}...")
    ys = run_delta_notch_sde(args,n_samples)
    
    noise_str = "{:0>3d}".format(int(noise_scale*1000))
    BASE_RESULTS_STR = f"{RESULT_DIR}/DN_traj_noise-{noise_str}_{{}}.txt"

    print(f"Writing results, noise={noise_scale:.4f}...")
    write_sde_results(ys,BASE_RESULTS_STR)
