import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    r1 = np.asarray(rater1, dtype=int)
    r2 = np.asarray(rater2, dtype=int)

    if r1.shape != r2.shape: 
        raise ValueError("Shape of rater 1 and rater 2 must match")

    
    
    n_classes = np.max((r1.max(), r2.max())) + 1

    # p_o
    p_o = np.mean(r1 == r2)

    # p_e
    p1 = np.bincount(r1, minlength=n_classes) / len(r1)
    p2 = np.bincount(r2, minlength=n_classes) / len(r2)
    p_e = np.sum(p1 * p2)

    if np.isclose(1 - p_e, 0):
        return 1.0
    
    return (p_o - p_e) / (1 - p_e) 