import numpy as np
def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    items = np.asarray(items, dtype=float)

    avg_rating = items[:, 0]
    num_votes = items[:, 1]
    denom = num_votes + min_votes
    result = (num_votes * avg_rating + min_votes * global_mean) / denom
    return result.tolist()