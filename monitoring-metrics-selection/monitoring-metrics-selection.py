import numpy as np

def get_confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp, tn, fp, fn

def accuracy(tp, tn, n):
    return (tp + tn) / n

def precision(tp, fp):
    denom = tp + fp 
    return tp / denom if denom > 0 else 0.0 

def recall(tp, fn):
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0 

def f1(pre, recall):
    denom = pre + recall
    return 2 * pre * recall / denom if denom > 0 else 0.0

def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def ranking(y_true: np.ndarray, y_pred: np.ndarray, k: int):
    idx = np.argsort(-y_pred)[:k]
    top_k_true = y_true[idx]

    relevant_in_top_k = np.sum(top_k_true == 1)
    total_relevant = np.sum(y_true == 1)

    precision_at_k = relevant_in_top_k / k 
    recall_at_k = (
        relevant_in_top_k / total_relevant 
        if total_relevant > 0 else 0.0
    )

    return precision_at_k, recall_at_k
    

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("True labels must be flatten")
    if y_pred.ndim != 1:
        raise ValueError("Predicted labels must be flatten")
    if y_true.shape != y_pred.shape:
        raise ValueError("True labels and predicted labels must have same shape")

    metrics = dict()
    if system_type == "classification":
        tp, tn, fp, fn = get_confusion_matrix_binary(y_true, y_pred)
        metrics["accuracy"] = accuracy(tp, tn, len(y_true))
        p = precision(tp, fp)
        r = recall(tp, fn)
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1'] = f1(p, r)
    elif system_type == "regression":
        metrics["mae"] = mae(y_true, y_pred)
        metrics['rmse'] = rmse(y_true, y_pred)    
    elif system_type == "ranking":
        metrics["precision_at_3"], metrics["recall_at_3"] = ranking(y_true, y_pred, 3)
    else:
        raise ValueError("system_type must be 'classification', 'regression' or 'ranking'")

    metrics = sorted(metrics.items())
    return metrics
        