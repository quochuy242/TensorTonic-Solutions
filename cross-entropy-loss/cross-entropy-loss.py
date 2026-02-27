import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    loss = []
    for true, pred in zip(y_true, y_pred):
        loss.append(-np.log(pred[true]))

    return np.mean(loss)