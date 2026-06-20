import numpy as np

def iou(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = 0
    union = 0

    rows, cols = y_true.shape

    for i in range(rows):
        for j in range(cols):

            true = y_true[i][j]
            pred = y_pred[i][j]

            if true or pred:
                union += 1  

                if true and pred:
                    intersection += 1

    if union == 0:
        return 1.0 

    return intersection / union

def mean_iou(y_trues, y_preds):

    total = 0
    n = len(y_trues)

    for i in range(n):
        total += iou(y_trues[i], y_preds[i])

    return total / n

def pixel_accuracy(y_trues, y_preds):

    tp = tn = fp = fn = 0

    for img_idx in range(len(y_trues)):

        y_true = y_trues[img_idx]
        y_pred = y_preds[img_idx]

        rows, cols = y_true.shape

        for i in range(rows):
            for j in range(cols):

                t = y_true[i][j]
                p = y_pred[i][j]

                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 1 and p == 0:
                    fn += 1
                else:
                    tn += 1

    total = tp + tn + fp + fn

    return (tp + tn) / total