import csv
import pathlib
from argparse import ArgumentParser
import numpy as np
import metrics
from error import TransformError


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean_features = np.mean(X, axis=0)
        self.std_features = np.std(X, axis=0)

    def transform(self, X) -> np.array:
        return (X - self.mean_features) / self.std_features


class OneHotEncoder:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.int8)
        self.dimensions = {i: np.unique(X[:, i]).size for i in np.arange(X.shape[1])}

    def transform(self, X) -> np.array:
        try:
            X = np.array(X, dtype=np.int8)
            X_transformed = np.eye(self.dimensions.get(0))[X[:, 0]]

            for i in np.arange(1, X.shape[1]):
                X_transformed = np.hstack((X_transformed, np.eye(self.dimensions.get(i))[X[:, i]]))
        except IndexError:
            raise TransformError('Category value should start with zero')
        return X_transformed


def get_path() -> str:
    parser = ArgumentParser(
        description='Implementation of linear models')
    parser.add_argument('--path', type=str, required=True, help='Path to files with data')
    return pathlib.Path(parser.parse_args().path)


def get_data(path, target_column: int, mapper=None) -> (np.array, np.array):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        matrix = np.asarray([get_mapped_row(row, mapper) for row in csv_reader])
    matrix = matrix.astype(np.float)

    full_columns = np.arange(matrix.shape[1])
    X_columns = full_columns[full_columns != [target_column]]
    return matrix[:, X_columns], matrix[:, target_column].flatten()


def get_mapped_row(row: dict, mapper=None) -> np.array:
    mapped_row = []
    
    for key, item in sorted(row.items()):
        if mapper is not None:
            column = mapper.get(key, key)
            if type(column) is not str:
                mapped_row.append(column.get(item))
            else:
                mapped_row.append(item)
        else:
            mapped_row.append(item)
    return np.array(mapped_row)


def classification_report(y_true, y_pred):
    print('--------------------------------')
    print('Accuracy -', metrics.accuracy(y_true, y_pred))
    print('Recall -', metrics.recall(y_true, y_pred))
    print('Precision -', metrics.precision(y_true, y_pred))
    print('F1 score -', metrics.f1_score(y_true, y_pred))
    print('--------------------------------')


def regression_report(y_true, y_pred):
    print('--------------------------------')
    print('MSE -', metrics.MSE(y_true, y_pred))
    print('RMSE -', metrics.RMSE(y_true, y_pred))
    print('MAN -', metrics.MAN(y_true, y_pred))
    print('--------------------------------')
