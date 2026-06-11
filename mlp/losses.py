import numpy as np

def cross_entropy_loss(y_pred, y_true):
    """
    y_pred: saídas da rede (após softmax)
    y_true: labels em formato one-hot encoding
    """
    samples = y_pred.shape[0]
    # Clip para evitar log(0)
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / samples
    return loss