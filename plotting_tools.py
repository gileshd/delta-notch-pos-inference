import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
import seaborn as sns


def make_joint_and_marginal_axes(marginal_spines_visible=False, figsize=(8,8)):
    """
    Create axes for a joint plot with and marginals.
    """
    # start with a square Figure
    fig = plt.figure(figsize=figsize)

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.0,
        hspace=0.0,
    )

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg1 = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg2 = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    if marginal_spines_visible is False:
        for ax in (ax_marg1, ax_marg2):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return ax_joint, ax_marg1, ax_marg2


def mesh_density(pdf, lims=(-6.5, 6.5), step=0.01, x_lims=None, y_lims=None):
    """
    Construct 2D meshgrid at points defined by lims and evaluate pdf
     at every point.
    """
    if x_lims is None:
        x_lims = lims
    if y_lims is None:
        y_lims = lims

    x, y = np.mgrid[x_lims[0] : x_lims[1] : step, y_lims[0] : y_lims[1] : step]
    pos = np.dstack((x, y))
    density = pdf(pos)
    return x, y, density


def plot_joint_density(
    μ, cov, lims, ax=None, legend=True, legend_kws={"loc": "upper left"}, step=0.01
):
    """ Plot density for bivariate normal with mean and covariance, μ, cov. """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    z1, z2 = [0, 1], [1, 0]

    pdf1 = lambda x: st.multivariate_normal.pdf(x, μ[z1], cov)
    x, y, density = mesh_density(pdf1, lims, step)
    cntr1 = ax.contour(x, y, density, colors=f"C0")

    pdf2 = lambda x: st.multivariate_normal.pdf(x, μ[z2], cov)
    x, y, density = mesh_density(pdf2, lims, step)
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
        label_base = r"$p(\mathbf{{g}}|\mathbf{{z}}={})$"
        label1 = label_base.format("\{1,2\}")
        label2 = label_base.format("\{2,1\}")
        ax.legend(
            [h1[0], h2[0]],
            [label1, label2],
            **legend_kws,
        )

    ax.set_xlabel("$g_A$")
    ax.set_ylabel("$g_B$")

    return ax


def plot_gauss_curves(μ, lims, colors=None, ax=None, orientation="horizontal"):
    """ Plot density for univarite normal with mean, μ. """

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if colors is None:
        colors = [f"C{n}" for n in range(len(μ))]

    g = np.linspace(*lims, 1000)
    for i, μi in enumerate(μ):
        f = st.norm.pdf(g, loc=μi)
        if orientation == "horizontal":
            ax.plot(g, f, color=colors[i], label=f"$z={i+1}$")
            ax.set_ylim(0.0, f.max() + 0.05)
        elif orientation == "vertical":
            ax.plot(f, g, color=colors[i])
            ax.set_xlim(0.0, f.max() + 0.05)
        else:
            raise ValueError("orientation must be either 'horizontal' or 'vertical'")

    return ax


def plot_joint_with_marginals(
    μ, cov, lims, legend=True, legend_kws={"loc": "upper left"}
):
    """ Plot joint and marginal densities for bivariate normal with mean and covariance, μ, cov. """

    ax_joint, ax_marg1, ax_marg2 = make_joint_and_marginal_axes()

    plot_joint_density(μ, cov, lims, ax=ax_joint, legend=legend, legend_kws=legend_kws)

    z1, z2 = [0, 1], [1, 0]
    plot_gauss_curves(μ[z1], lims, ax=ax_marg1)
    plot_gauss_curves(μ[z2], lims, ax=ax_marg2, orientation="vertical")

    return ax_joint, ax_marg1, ax_marg2


# TODO: Add option to scale alpha of contours.
def plot_joint_and_marginal_kde(Y, axs, t, z):
    """ Plot joint and marginal kde for bivariate data, Y."""

    ls = ["dashed", "solid"][t]

    joint_plot_kwargs = {
        "linestyles": ls,
        "linewidths": [1.3, 1.3][t],
        "levels": [8, 6][t],
        "thresh": [0.15, 0.05][t],
    }

    marginal_plot_kwargs = {"ls": ls}

    kde_plot_kwargs = {
        "color": f"C{z}",
        "bw_adjust": [1, 0.25][t],
    }

    z_label = ["[1,2]", "[2,1]"][z]
    label = f"$\\mathbf{{z}}={z_label}$, $t={t}$"

    g1, g2 = Y.T
    ax_joint, ax_marg1, ax_marg2 = axs

    sns.kdeplot(x=g1, y=g2, ax=ax_joint, **joint_plot_kwargs, **kde_plot_kwargs)
    sns.kdeplot(x=g1, ax=ax_marg1, **marginal_plot_kwargs, **kde_plot_kwargs, label=label)
    sns.kdeplot(y=g2, ax=ax_marg2, **marginal_plot_kwargs, **kde_plot_kwargs)

