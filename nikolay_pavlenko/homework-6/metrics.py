import torch


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / y_true.shape[0]


def mape(y_true, y_pred):
    return 100 * torch.abs((y_true - y_pred) / y_true).sum() / y_true.shape[0]


def mpe(y_true, y_pred):
    return 100 * ((y_true - y_pred) / y_true).sum() / y_true.shape[0]


def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return torch.abs(y_true - y_pred).sum() / y_true.shape[0]


def r2(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().type(torch.float32) / y_true.shape[0]


def recall(y_true, y_pred):
    return (y_true.type(torch.int32) & y_pred.type(torch.int32)).sum() / y_true.sum()


def precision(y_true, y_pred):
    return (y_true.type(torch.int32) & y_pred.type(torch.int32)).sum() / y_pred.sum()


def fbeta(y_true, y_pred, beta):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return (1 + beta ** 2) * (prec * rec) / ((beta ** 2) * prec + rec)


def f1(y_true, y_pred):
    return fbeta(y_true, y_pred, 1)
