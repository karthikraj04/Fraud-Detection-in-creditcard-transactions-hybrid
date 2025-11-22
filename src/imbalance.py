from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(y):
    """Compute balanced class weights for fraud dataset."""

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    return {
        0: class_weights[0],
        1: class_weights[1]
    }
