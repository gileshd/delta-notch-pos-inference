import numpy as np

def drop_nans(x):
    nan_mask = np.isnan(x)
    return x[~nan_mask]


def argmax_ij(X):
    idx = np.argmax(X)
    i = idx // X.shape[1]
    j = idx % X.shape[1]
    return int(i),int(j)



