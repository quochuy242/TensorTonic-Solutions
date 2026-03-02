import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    return {
        'min': np.full(shape=D, fill_value=np.inf), 
        'max': np.full(shape=D, fill_value=-np.inf)
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    X_batch = np.asarray(X_batch, dtype=float)

    if X_batch.ndim == 1:
        X_batch = X_batch.reshape(1, -1)

    if state['min'] is None:
        state = streaming_minmax_init(X_batch.shape[1])
        
    batch_min, batch_max = np.min(X_batch, axis=0), np.max(X_batch, axis=0)
 
    state['min'] = np.minimum(state['min'], batch_min)
    state['max'] = np.maximum(state['max'], batch_max)

    diff_vals = state['max'] - state['min']
    diff_vals = np.where(diff_vals < eps, 1.0, diff_vals)

    X_norm = (X_batch - state['min']) / diff_vals
    return X_norm 
    