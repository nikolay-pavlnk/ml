import numpy as np


class Stacked:
    def predict(self, models: list, x: list):
        return np.mean([models[1].predict(x[1])[0], models[0](x[0]).item()])


class LabelEncoder:
    def fit(self, X: np.array):
        unique = np.unique(X)
        self.labels = {
            i: dict(zip(np.unique(X[:, i]), np.arange(np.unique(X[:, i].size))))
            for i in np.arange(X.shape[1])
        }

    def transform(self, X: np.array):
        to_return = np.array(
            [
                [self.labels[key].get(value, 999999) for value in X[:, key]]
                for key in self.labels
            ]
        )
        return to_return.T

    def fit_transform(self, X: np.array):
        self.fit(X)
        return self.transform(X)
