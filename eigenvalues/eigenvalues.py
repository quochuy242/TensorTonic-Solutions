import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here

    # None case 
    if not matrix: 
        return None

    try:
        matrix = np.asarray(matrix, dtype=float)
    except:
        return None 
    
    # Non-square case
    if matrix.ndim == 1: 
        return None
    if matrix.shape[0] != matrix.shape[1]: 
        return None 
    
    eigenvalues = np.linalg.eigvals(matrix)

    idx = np.lexsort((eigenvalues.real, ))
    return eigenvalues[idx]
    
    
    

    