import numpy as np
from sde_systems.delta_notch_tools import load_sde_results, write_sde_results

RESULT_DIR = "results/noise_experiments"
OUT_DIR = "results/noise_experiments/delta_delta_t0t1"

noise_range = np.linspace(0,0.25,11)

for noise_scale in noise_range:
    noise_str = "{:0>3d}".format(int(noise_scale*1000))
    BASE_RESULTS_STR = f"{RESULT_DIR}/DN_traj_noise-{noise_str}_{{}}.txt"
    print(f"Loading results noise={noise_scale:.4f}")
    ys_load = load_sde_results(BASE_RESULTS_STR, genes=["delta1", "delta2"])
    ys_t0t1 = ys_load[:,[0,-1],:]
    
    BASE_OUT_STR = f"{OUT_DIR}/DD_t0t1_noise-{noise_str}_{{}}.txt"
    print(f"Writing results noise={noise_scale:.4f}")
    write_sde_results(ys_t0t1,BASE_OUT_STR)
