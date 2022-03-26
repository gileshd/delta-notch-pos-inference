import numpy as np
from matplotlib import pyplot as plt

plt.style.use("thesis")

OUT_FILE = "../SDE_figures/sigmoid_plot.pdf"

def sigmoid(x,k):
    return 1/(1 + np.exp(-k*x))

fig, ax = plt.subplots()

x = np.linspace(-1,1,1000)
ax.plot(x,sigmoid(x,20))

ax.set_xlabel("$x$")
ax.set_ylabel(r"$\mathcal{S}(x)$")

plt.savefig(OUT_FILE,bbox_inches="tight")
