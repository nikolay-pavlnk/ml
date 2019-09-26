import numpy as np


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))


def MAN(y_true, y_pred):
    return np.mean(np.fabs(y_true - y_pred))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def recall(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return TP / (TP + FN)


def precision(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    return TP / (TP + FP)


def f1_score(y_true, y_pred):
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)    
    return 2 * rec * prec / (rec + prec)
