import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    is_1d = False 
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        is_1d = True

    if strategy == 'mean':
        fill_values = np.nanmean(X, axis=0)
    elif strategy == 'median':
        fill_values = np.nanmedian(X, axis=0)
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    fill_values = np.where(np.isnan(fill_values), 0, fill_values)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = fill_values[cols]
    
    return X.flatten() if is_1d else X

    
    