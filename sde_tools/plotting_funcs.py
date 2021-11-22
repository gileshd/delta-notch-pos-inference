from matplotlib import pyplot as plt

def plot_samples(
    ts,
    samples,
    ax=None,
    xlabel="$t$",
    ylabel="$Y_t$",
    plot_mean=False,
    legend=False,
    **kwargs
):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()

    if ax is None:
        fig, ax = plt.subplots()

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.3
    if 'color' not in kwargs:
        kwargs['color'] = "C0"

    for i, sample in enumerate(samples):
        ax.plot(ts, sample, **kwargs)

    if plot_mean:
        sample_mean = samples.mean(0)
        ax.plot(ts, sample_mean, color="k", label="sample mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    return ax
