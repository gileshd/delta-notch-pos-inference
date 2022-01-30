import numpy as np

def drop_nans(x):
    nan_mask = np.isnan(x).any((1,2))
    return x[~nan_mask]



