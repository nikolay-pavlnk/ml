import numpy as np


class OneHotEncoder:
    def __init__(self):
        pass

    def fit(self, X: np.array, y=None):
        X = np.array(X, dtype=np.int8)
        self.dimensions = {i: np.unique(X[:, i]).size for i in np.arange(X.shape[1])}

    def transform(self, X) -> np.array:
        try:
            X = np.array(X, dtype=np.int8)
            X_transformed = np.eye(self.dimensions.get(0))[X[:, 0]]

            for i in np.arange(1, X.shape[1]):
                X_transformed = np.hstack(
                    (X_transformed, np.eye(self.dimensions.get(i))[X[:, i]])
                )
        except IndexError:
            raise IndexError("Category value should start with zero")
        return X_transformed


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X: np.array, y=None):
        self.mean_features = np.mean(X, axis=0)
        self.std_features = np.std(X, axis=0)

    def transform(self, X: np.array) -> np.array:
        return (X - self.mean_features) / self.std_features


def train_test_split(
    X: np.array, y: np.array, task: str, test_size=0.2, random_state=42
):
    np.random.seed(random_state)

    if task == "regression":
        n = int(y.size * test_size)
        shuffled_ind = np.random.permutation(y.size)
        return (
            X[shuffled_ind[n:]],
            X[shuffled_ind[:n]],
            y[shuffled_ind[n:]],
            y[shuffled_ind[:n]],
        )

    elif task == "classification":
        train_ind, test_ind = list(), list()
        for label in np.unique(y):
            label_indexes = np.nonzero(y == label)[0]
            np.random.shuffle(label_indexes)

            n = int(test_size * label_indexes.size)

            train_ind.extend(label_indexes[n:].tolist())
            test_ind.extend(label_indexes[:n].tolist())

        np.random.shuffle(test_ind)
        np.random.shuffle(train_ind)
        return X[train_ind], X[test_ind], y[train_ind], y[test_ind]
