import numpy as np
import math 

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    factorial = np.vectorize(math.factorial)
    
    pmf = np.exp(-lam) * np.power(lam, k) / factorial(k)
    cdf = np.sum([
        np.exp(-lam) * np.power(lam, i) / factorial(i)
        for i in range(k + 1)
    ])

    return pmf, cdf 