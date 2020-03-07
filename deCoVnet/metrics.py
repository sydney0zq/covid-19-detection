import numpy as np

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out

def precision_recall(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    # import pdb; pdb.set_trace()
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

def sensitivity_specificity(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    N = len(y_score)
    tp, fp = 0, 0
    condition_positive, condition_negative = np.sum(y_true), N-np.sum(y_true)

    sensitivity, specificity = np.zeros(N), np.zeros(N)

    for i in range(N):
        predicted_positive = i+1
        predicted_negative = N - predicted_positive
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        tn = condition_negative - fp

        sensitivity[i] = tp / float(condition_positive)
        specificity[i] = tn / float(condition_negative + 1e-6)

        # print( "tp: {}, fp: {}, tn: {}, sens: {}, spec: {}".format( tp,fp,tn,sensitivity[i], specificity[i]  )  )

    sensitivity, specificity = np.r_[0, sensitivity, 1], np.r_[1, specificity, 0]
    auc = 0
    for i in range(len(sensitivity)-1):
        # auc += (sensitivity[i+1]-sensitivity[i]) * specificity[i]
        auc += (sensitivity[i+1]-sensitivity[i]) * specificity[i]

    return sensitivity, specificity, auc


def naivepr(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    N = len(y_score)
    tp, fp = 0, 0
    condition_positive, condition_negative = np.sum(y_true), N-np.sum(y_true)

    precision, recall = np.zeros(N), np.zeros(N)

    for i in range(N):
        predicted_positive = i+1
        predicted_negative = N - predicted_positive
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        tn = condition_negative - fp

        precision[i] = tp / float(predicted_positive)
        recall[i] = tp / float(condition_positive)

    precision, recall = np.r_[1, precision, 0], np.r_[0, recall, 1]
    auc = 0
    for i in range(len(recall)-1):
        auc += (recall[i+1]-recall[i]) * precision[i]

    return precision, recall, auc



