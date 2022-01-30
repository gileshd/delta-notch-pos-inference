import numpy as np
from matplotlib import pyplot as plt


def sample_y0_DN(μ1, μ2, cov, n_samples, notch=0.5):
    y0_delta = np.random.multivariate_normal(
        [μ1, μ2], cov, size=n_samples
    )
    y0_notch = np.ones((n_samples, 2)) * notch
    return np.hstack((y0_delta, y0_notch))


def sample_y0_DN_mixed(μ1, μ2, cov, batch_size, notch=0.5):
    batch1 = sample_y0_DN(μ1, μ2, cov, batch_size, notch)
    batch2 = sample_y0_DN(μ2, μ1, cov, batch_size, notch)
    return np.vstack((batch1, batch2))


def plot_DN_state_space(ys,ax=None, **plot_kwargs):
    """
    ys (samples x time x 4)
    """
    d1,d2 = ys[...,:2].T

    if ax is None:
        fig, ax = plt.subplots()

    base_kwargs = {"color":"C0", "alpha":0.3}
    base_kwargs.update(plot_kwargs)
    ax.plot(d1,d2,**base_kwargs);

    ax.axis('scaled');
    lim_max = max(d1.max(),d2.max(),ax.get_xlim()[1],ax.get_ylim()[1])
    lims = (0,lim_max)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)

    return ax
