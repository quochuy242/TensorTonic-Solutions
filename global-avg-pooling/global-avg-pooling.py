import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    if x.ndim not in (3, 4):
        raise ValueError("Support (C,H,W) and (N,C,H,W)")
    
    H, W = x.shape[-2:]
    result = x.sum(axis=(-2, -1)) / (H * W)
    
    return result
    
    