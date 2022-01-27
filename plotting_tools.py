import numpy as np
from matplotlib import pyplot as plt


def mesh_density(pdf, lims=(-6.5, 6.5), x_lims=None, y_lims=None):
    if x_lims is None:
        x_lims = lims
    if y_lims is None:
        y_lims = lims

    x, y = np.mgrid[x_lims[0] : x_lims[1] : 0.01, y_lims[0] : y_lims[1] : 0.01]
    pos = np.dstack((x, y))
    density = pdf(pos)
    return x, y, density


def plot_joint_density(mean1, mean2, cov, lims, ax=None, legend=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    pdf1 = lambda x: st.multivariate_normal.pdf(x, mean1, cov)
    x, y, density = mesh_density(pdf1, lims)
    cntr1 = ax.contour(x, y, density, colors=f"C0")

    pdf2 = lambda x: st.multivariate_normal.pdf(x, mean2, cov)
    x, y, density = mesh_density(pdf2, lims)
    cntr2 = ax.contour(x, y, density, colors=f"C1")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.axis("scaled")

    tick_labels = [r"$\mu_1$", r"$\mu_2$"]
    ax.set_xticks(μ)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(μ)
    ax.set_yticklabels(tick_labels)

    if legend:
        h1, _ = cntr1.legend_elements()
        h2, _ = cntr2.legend_elements()
        ax.legend([h1[0], h2[0]], [r"$z=\{1,2\}$", r"$z=\{2,1\}$"])

    ax.set_xlabel("$g_1$")
    ax.set_ylabel("$g_2$")

    return ax


def plot_gauss_curves(μ, lims, ax=None, orientation="horizontal"):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    g = np.linspace(*lims, 1000)
    for i, μi in enumerate(μ):
        f = st.norm.pdf(g, loc=μi)
        if orientation == "horizontal":
            ax.plot(g, f)
        elif orientation == "vertical":
            ax.plot(f, g)
        else:
            raise ValueError("orientation must be either 'horizontal' or 'vertical'")

    return ax
