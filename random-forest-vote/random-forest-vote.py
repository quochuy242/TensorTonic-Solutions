import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    preds = np.asarray(predictions, dtype=int)

    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    n_samples = preds.shape[1]

    result = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        values, counts = np.unique(preds[:, i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    
    return result.tolist()