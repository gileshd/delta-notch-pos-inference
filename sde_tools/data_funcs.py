import numpy as np

def drop_nans(x):
    nan_mask = np.isnan(x)
    return x[~nan_mask]



