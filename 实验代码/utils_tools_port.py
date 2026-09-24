"""Port of E:/mttta/ML-TTA/utils/tools.py::mAP (official continuous AP)."""
import numpy as np


def mAP_official(targets, preds):
    epsilon = 1e-8
    aps = []
    for k in range(preds.shape[1]):
        scores, target = preds[:, k], targets[:, k]
        indices = scores.argsort()[::-1]
        total_count_ = np.cumsum(np.ones((len(scores), 1)))
        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        aps.append(np.sum(pp) / (total + epsilon))
    return float(100 * np.mean(aps))
