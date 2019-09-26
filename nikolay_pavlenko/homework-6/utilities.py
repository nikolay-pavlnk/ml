import numpy as np
import time
from torch.utils.data import Dataset

import metrics
from model_selection import OneHotEncoder, StandardScaler


def isvalid(arr: np.array) -> bool:
    try:
        return bool(np.float(arr) + 1)
    except ValueError:
        return False


isnumber = np.vectorize(isvalid)


class LoadDataset(Dataset):
    def one_hot_fit(self, X):
        self.encoder = OneHotEncoder()
        self.encoder.fit(X)

    def scaler_fit(self, X):
        self.scaler = StandardScaler()
        self.scaler.fit(X)

    def scaler_transform(self, X: np.array) -> np.array:
        return self.scaler.transform(X)

    def one_hot_transform(self, X: np.array) -> np.array:
        return self.encoder.transform(X)

    def transform(self, csvreader):
        X, self.y = csvreader.get_X_y()
        X_one_hot = self.one_hot_transform(X[:, csvreader.categorical_idx])
        X_new = np.hstack((X_one_hot, X[:, csvreader.continuous_idx]))
        self.X = self.scaler_transform(X_new)
        self.len = self.X.shape[0]
        self.dim = self.X.shape[1]

    def fit(self, csvreader):
        X, _ = csvreader.get_X_y()
        self.one_hot_fit(X[:, csvreader.categorical_idx])
        X_one_hot = self.one_hot_transform(X[:, csvreader.categorical_idx])
        X_new = np.hstack((X_one_hot, X[:, csvreader.continuous_idx]))
        self.scaler_fit(X_new)

    def __getitem__(self, index: np.array) -> np.array:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.len

    def __dim__(self) -> int:
        return self.dim


class CsvReader:
    def __init__(self, target: str, path: str, n_unique: int):
        self.__target = target
        self.__path = path
        self.__n_unique = n_unique

    def drop_columns(self, data: np.array):
        col_to_drop = list()
        for i in self.__continuous_idx:
            if not isnumber(data[1:, i]).any():
                col_to_drop.append(i)
            elif np.unique(data[1:, i]).size == data[1:, i].size:
                col_to_drop.append(i)

        data = np.delete(data, col_to_drop, axis=1)
        self.__continuous_idx = np.setdiff1d(
            self.__continuous_idx, col_to_drop
        ).tolist()

    def fill_nan(self, data: np.array):
        for i in np.arange(data.shape[1]):
            column = data[1:, i]
            idx_nan = np.nonzero(column == "")[0]
            idx_val = np.nonzero(column != "")[0]
            if isnumber(column).any():
                temp_column = np.array(column[idx_val], dtype=np.float)
                column[idx_nan] = np.median(temp_column)
            else:
                values, counts = np.unique(column, return_counts=True)
                column[idx_nan] = values[np.argmax(counts)]

        return data

    def __read_data(self):
        data = np.genfromtxt(self.__path, dtype=str, delimiter=",")
        return self.fill_nan(data)

    def __get_indexes(self, data):
        features_name = data[0].tolist()
        self.__target_index = features_name.index(self.__target)
        full_idx = np.arange(len(features_name)).tolist()
        idx_without_target = set(full_idx) - set([self.__target_index])
        return idx_without_target, features_name

    def __set_new_features_name(self, feature_names: list):
        temp_names = np.array(feature_names)
        self.feature_names = (
            temp_names[self.__categorical_idx].tolist()
            + temp_names[self.__continuous_idx].tolist()
        )
        self.categorical_idx = np.arange(len(self.__categorical_idx))
        self.continuous_idx = np.arange(
            len(self.__categorical_idx),
            len(self.__continuous_idx) + len(self.__categorical_idx),
        )

    def get_X_y(self):
        data = self.__read_data()
        idx_without_target, features_name = self.__get_indexes(data)
        self.__set_feature_type(data, idx_without_target, features_name)
        self.drop_columns(data)
        self.__set_new_features_name(features_name)

        N = data[1:].shape[0]
        X_categorical = np.zeros((N, len(self.__categorical_idx)))
        X_continuos = np.zeros((N, len(self.__continuous_idx)))
        for i, j in enumerate(self.__categorical_idx):
            X_categorical[:, i] = np.unique(data[1:, j], return_inverse=True)[1]

        for i, j in enumerate(self.__continuous_idx):
            X_continuos[:, i] = np.array(data[1:, j], dtype=np.float)

        X = np.hstack((X_categorical, X_continuos))
        y = np.ascontiguousarray(data[1:, self.__target_index], dtype=np.float)
        return X, y

    def __set_feature_type(
        self, data: np.array, idx_without_target: list, features_name: list
    ):

        self.__categorical_idx, self.__continuous_idx = list(), list()
        self.features_categories = dict()

        for i in idx_without_target:
            if np.unique(data[1:, i]).shape[0] <= self.__n_unique:
                if data[1:, i][0].isalpha():
                    self.features_categories[features_name[i]] = np.unique(
                        data[1:, i]
                    ).tolist()
                    self.__categorical_idx.append(i)
                else:
                    col_fl = np.array(data[1:, i], dtype=np.float)
                    col_int = np.array(data[1:, i], dtype=np.int)
                    if np.sum(col_fl - col_int) == 0:
                        self.features_categories[features_name[i]] = np.unique(
                            data[1:, i]
                        ).tolist()
                        self.__categorical_idx.append(i)

        self.__continuous_idx = list(idx_without_target - set(self.__categorical_idx))


def classification_report(y_true, y_pred):
    print("Accuracy", metrics.accuracy(y_true, y_pred).item())
    print("Recall", metrics.recall(y_true, y_pred).item())
    print("Precision", metrics.precision(y_true, y_pred).item())
    print("F1", metrics.f1(y_true, y_pred).item())


def regression_report(y_true, y_pred):
    print("MSE", metrics.mse(y_true, y_pred).item())
    print("RMSE", metrics.rmse(y_true, y_pred).item())
    print("MAPE", metrics.mape(y_true, y_pred).item())
    print("MPE", metrics.mpe(y_true, y_pred).item())
    print("R2", metrics.r2(y_true, y_pred).item())


def measure_time(func):
    def _wrapper(_, X, y):
        start_time = time.time()
        result = func(_, X, y)
        print("Execution time: ", time.time() - start_time)
        return result

    return _wrapper
