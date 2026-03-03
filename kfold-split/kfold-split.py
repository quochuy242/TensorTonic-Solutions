import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    # Shuffle step
    if rng is None:
        rng = np.random.default_rng()
        
    indices = np.arange(N, dtype=int)
    if shuffle:
        rng.shuffle(indices)
    
    # Split into k folds
    folds = np.array_split(indices, k)

    # Split to train & val
    splits = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([
            folds[j] for j in range(k) if j != i
        ])
        splits.append((train_idx, val_idx))

    return splits
    
    

    
    
    