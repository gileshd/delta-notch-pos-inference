import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt

plt.style.use('thesis')

OUT_FILE = "SDE_figures/DN_noise_performance.pdf"

noise_lims = (0.0,0.25)
noise_range = np.linspace(*noise_lims, 101)

p_marg = 0.9
bayes_opt = st.norm.cdf(st.norm.ppf(p_marg)*np.sqrt(2))
ind_perf = p_marg**2

NOISE_FILE = "../results/noise_experiments/noise_perf_000-025_101.txt"
noise_perf = np.loadtxt(NOISE_FILE)

plt.figure(figsize=(10,6))

plt.hlines(bayes_opt,*noise_lims,ls='--',color="C5",label="Optimal Joint Performace");
plt.plot(noise_range,noise_perf,color="C0",label="Independent Joint Performance $t=1$");
plt.hlines(p_marg,*noise_lims,ls='dotted',color="grey", label="Marginal Performance $t=0$");
plt.hlines(ind_perf,*noise_lims,ls='-.',color="grey",label="Idependent Joint Performance $t=0$");

plt.legend(loc="lower left");
plt.xlim(*noise_lims);
plt.ylim(0.5,1);
plt.xlabel("Noise scale");
plt.ylabel("Performance");

plt.savefig(OUT_FILE,bbox_inches="tight")
