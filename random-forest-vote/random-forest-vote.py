import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    preds = np.asarray(predictions, dtype=int)

    if preds.ndim == 1:
        preds = preds[None, :]

    n_classes = preds.max() + 1

    votes = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_classes).argmax(),
        axis=0,
        arr=preds 
    )

    return votes.tolist()