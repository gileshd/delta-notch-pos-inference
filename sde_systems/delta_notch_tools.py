import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def drop_nans_DN(x):
    nan_mask = np.isnan(x).any((1, 2))
    return x[~nan_mask]


def calculate_y0_std(μ1, μ2, p=0.90):
    """
    Calculate the std required to give a marginal cdf of 0.9
     at (μ1 - μ2)/2.
    """
    d = (μ2 - μ1) / 2
    τ = st.norm.ppf(p)
    return d / τ


def sample_y0_DN(μ1, μ2, cov, n_samples, notch=0.5):
    """
    Sample y0 for Delta-Notch system with dA, dB ~ N([μ1,μ2],cov)
     and nA, nB = 0.5.
    """
    y0_delta = np.random.multivariate_normal([μ1, μ2], cov, size=n_samples)
    y0_notch = np.ones((n_samples, 2)) * notch
    return np.hstack((y0_delta, y0_notch))


def sample_y0_DN_mixed(μ1, μ2, cov, batch_size, notch=0.5):
    """
    Sample two batches of y0 for Delta-Notch system with z = {A,B}, {B,A}.
    """
    batch1 = sample_y0_DN(μ1, μ2, cov, batch_size, notch)
    batch2 = sample_y0_DN(μ2, μ1, cov, batch_size, notch)
    return np.vstack((batch1, batch2))


def optimal_threshold_marg_perf(dA, dB):
    """
    Calculate the marginal performance of the optimal threshold to
     distinguish between the distribution of values in dA, dB.
    """
    τ = np.linspace(0, 1, 1000)
    sorted_idx = np.argsort(np.array([np.median(dA), np.median(dB)]))
    d1, d2 = np.array([dA, dB])[sorted_idx]
    B1 = d1[:, None] < τ[None, :]
    B2 = d2[:, None] > τ[None, :]
    p_marg_av = (B1.mean(0) + B2.mean(0)) / 2
    return p_marg_av.max()


def optimal_threshold_joint_perf(dA, dB):
    """
    Calculate the joint performance of the optimal threshold to
     distinguish between the distribution of values in dA, dB.
    """
    τ = np.linspace(0, 1, 1000)
    sorted_idx = np.argsort(np.array([np.median(dA), np.median(dB)]))
    d1, d2 = np.array([dA, dB])[sorted_idx]
    B = (d1[:, None] < τ[None, :]) & (d2[:, None] > τ[None, :])
    return B.mean(0).max()


def write_sde_results(ys, RESULTS_FILE_BASE):
    """ Write sde results to 4 txt files. """
    n_genes = ys.shape[-1]
    genes = ["deltaA", "deltaB", "notchA", "notchB"][:n_genes]
    for n, g in enumerate(genes):
        GENE_RESULTS_FILE = RESULTS_FILE_BASE.format(g)
        np.savetxt(GENE_RESULTS_FILE, ys[:, :, n], fmt="%.6f")


def load_sde_results(RESULTS_FILE_BASE, genes = ["delta1", "delta2", "notch1", "notch2"]):
    """ Load sde results from 4 text files. """
    result_list = []
    for n, g in enumerate(genes):
        GENE_RESULTS_FILE = RESULTS_FILE_BASE.format(g)
        result_list.append(np.loadtxt(GENE_RESULTS_FILE))
    return np.dstack(result_list)


def plot_DN_state_space(ys, ax=None, **plot_kwargs):
    """
    ys (samples x time x 4)
    """
    d1, d2 = ys[..., :2].T

    if ax is None:
        fig, ax = plt.subplots()

    base_kwargs = {"color": "C0", "alpha": 0.3}
    base_kwargs.update(plot_kwargs)
    ax.plot(d1, d2, **base_kwargs)

    ax.axis("scaled")
    lim_max = max(d1.max(), d2.max(), ax.get_xlim()[1], ax.get_ylim()[1])
    lims = (0, lim_max)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

    return ax


def plot_delta_notch_time_course(y, ax=None):
    gene_names = ["delta1", "delta2", "notch1", "notch2"]
    gene_label_dict = {
        "delta1": r"$\mathrm{Delta_A}$",
        "delta2": r"$\mathrm{Delta_B}$",
        "notch1": r"$\mathrm{Notch_A}$",
        "notch2": r"$\mathrm{Notch_B}$",
    }

    gene_color_dict = {
        "delta1": "#054d80",
        "delta2": "#6597b8",
        "notch1": "#ad0c00",
        "notch2": "#de7f78",
    }

    if ax is None:
        fig, ax = plt.subplots()

    t = np.linspace(0, 1, len(y))

    for n, g in enumerate(y.T):
        gene = gene_names[n]
        ax.plot(t, g, color=gene_color_dict[gene], label=gene_label_dict[gene])

    ax.set_xlim(0, 1)
    ylims = ax.get_ylim()
    ax.set_ylim(0, ylims[1])

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Expression")


def plot_delta_delta_time_course(y):
    d1, d2 = y[..., :2].T

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([d1, d2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    color = np.linspace(0, 1, len(d1))

    lc = LineCollection(segments, cmap="viridis")
    lc.set_array(color)
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, fraction=0.05, shrink=0.9, pad=-0.2, label="Time")

    ax.axis("scaled")
    lim_max = max(d1.max(), d2.max())
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    ax.set_xlabel(r"$\mathrm{Delta_A}$")
    ax.set_ylabel(r"$\mathrm{Delta_B}$")


def plot_delta_delta_time_course_multi(ys):

    ys = drop_nans_DN(ys)

    fig, ax = plt.subplots()

    for y in ys:
        d1, d2 = y[..., :2].T

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([d1, d2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        color = np.linspace(0, 1, len(d1))

        lc = LineCollection(segments, cmap="viridis")
        lc.set_array(color)
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        
    fig.colorbar(line, ax=ax, fraction=0.05, shrink=0.9, pad=-0.2, label="Time")

    ax.axis("scaled")
    lim_max = max(ys[...,0].max(), ys[...,1].max())
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    ax.set_xlabel(r"$\mathrm{Delta_A}$")
    ax.set_ylabel(r"$\mathrm{Delta_B}$")
