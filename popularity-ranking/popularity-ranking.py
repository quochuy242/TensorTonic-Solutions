import numpy as np
def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    items = np.asarray(items, dtype=float)

    R = items[:, 0]
    v = items[:, 1]
    m = min_votes
    C = global_mean
    denom = v + m
    return ((v * R + m * C) / denom).tolist()