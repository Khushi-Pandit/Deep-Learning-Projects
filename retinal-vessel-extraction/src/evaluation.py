import numpy as np

def compute_metrics(pred, gt):
    pred = pred // 255
    gt = gt // 255

    TP = np.sum((pred == 1) & (gt == 1))
    FN = np.sum((pred == 0) & (gt == 1))

    sensitivity = TP / (TP + FN + 1e-8)
    return sensitivity