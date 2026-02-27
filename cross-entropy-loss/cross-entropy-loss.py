import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_pred) != len(y_true):
        raise ValueError("Num of samples must match")

    eps = 1e-8 
    y_pred = np.clip(y_pred, eps, 1 - eps)

    correct_class_probs = y_pred[np.arange(len(y_true)), y_true]

    loss = - np.mean(np.log(correct_class_probs))

    return loss 