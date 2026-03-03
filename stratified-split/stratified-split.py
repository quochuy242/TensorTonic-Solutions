import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    train_idx = []
    test_idx = []
    
    class_names, class_counts = np.unique(y, return_counts=True)

    for class_name in class_names:
        indices = np.where(y == class_name)[0]

        rng.shuffle(indices) if rng else np.random.shuffle(indices)

        n_test = int(round(len(indices) * test_size))

        test_idx.append(indices[:n_test])
        train_idx.append(indices[n_test:])

    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )