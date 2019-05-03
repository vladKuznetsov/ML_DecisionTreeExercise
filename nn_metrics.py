
import numpy as np


def compareBoolPredictions(clusters : int, y_pred : np.ndarray, y_labels : np.ndarray):

    retVal = np.zeros([clusters, 3])

    if  not len(y_pred) == len(y_labels):
        print(1426, "Lengths are different. Takes the minimum")

    nn = min(len(y_pred), len(y_labels))

    for i in range(nn):
        retVal[y_pred[i], 0] += 1

        if y_labels[i] >   0:        retVal[y_pred[i],1] += 1
        if y_labels[i] <=  0:        retVal[y_pred[i],2] += 1

    return retVal