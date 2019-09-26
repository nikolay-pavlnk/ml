import torch
from utilities import measure_time


class LinearRegression:
    def __init__(
        self,
        learning_rate=10e-3,
        weight_dist=10e-4,
        epochs=10e3,
        intercept=True,
        normal_equation=False,
        alpha=1,
        penalty="l2",
        batch_size=64,
        random_state=17,
    ):
        self.learning_rate = learning_rate
        self.weight_dist = weight_dist
        self.epochs = epochs
        self.intercept = intercept
        self.normal_equation = normal_equation
        self.alpha = alpha
        self.penalty = penalty
        self.batch_size = batch_size
        self.random_state = random_state
        self.w = None

    def __calculate_gradient(self, X, y):
        if self.penalty == "l2":
            regulator = torch.cat([self.w[0][None], self.w[1:] * self.alpha])
        elif self.penalty == "l1":
            regulator = torch.cat(
                [self.w[0][None], torch.sign(self.w[1:]) * self.alpha]
            )

        return torch.mv(X.t(), torch.mv(X, self.w) - y) + regulator

    def __add_intercept(self, X):
        return torch.cat([X.new_ones(X.size()[0], 1), X], dim=1)

    def __calculate_loss(self, X, y):
        if self.penalty == "l2":
            reg_loss = self.alpha * torch.dot(self.w, self.w).item()
        elif self.penalty == "l1":
            reg_loss = self.alpha * torch.sum(torch.abs(self.w)).item()
        return torch.mean((y - torch.mv(X, self.w)) ** 2 + reg_loss).item()

    def __get_analytical_solution(self, X, y):
        if self.penalty == "l1":
            raise AttributeError("Solution with lasso regulator does not exist")

        if self.alpha == 0 and self.penalty == "l2":
            print("WARNING: Ridge coefficient is equal to zero")

        input_1 = torch.inverse(torch.mm(X.t(), X)) + self.alpha * torch.eye(
            X.size()[1]
        )
        input_2 = torch.mv(X.t(), y)
        self.w = torch.mv(input_1, input_2)
        return self

    @measure_time
    def fit(self, X, y):
        if self.intercept:
            X = self.__add_intercept(X)

        if self.normal_equation:
            return self.__get_analytical_solution(X, y)

        self.w = X.new_zeros(X.shape[1])
        self.loss = [self.__calculate_loss(X, y)]
        w_old = self.w.clone()
        w_dist = 1 ** 10
        n_iter = 0
        N = X.size()[0]
        torch.manual_seed(self.random_state)

        while (n_iter < self.epochs) and (w_dist > self.weight_dist):
            indices = torch.randperm(X.size()[0])
            X, y = X[indices], y[indices]

            for i in range(0, N, self.batch_size):
                X_train, y_train = (
                    X[i : i + self.batch_size],
                    y[i : i + self.batch_size],
                )
                self.w -= self.learning_rate * self.__calculate_gradient(
                    X_train, y_train
                )
                self.loss.append(self.__calculate_loss(X_train, y_train))

                w_dist = torch.norm(self.w - w_old).item()
                w_old = self.w.clone()
                n_iter += 1

        return self

    def predict(self, X):
        if self.intercept:
            X = self.__add_intercept(X)

        if self.w is None:
            raise AttributeError("Not fitted yet")

        return torch.mv(X, self.w)


class LogisticRegression:
    def __init__(
        self,
        C=1,
        penalty="l2",
        learning_rate=0.5,
        weight_dist=10e-4,
        epochs=10e5,
        intercept=True,
        random_state=17,
        threshold=0.5,
        tracker=None,
        batch_size=64,
    ):
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.weight_dist = weight_dist
        self.epochs = epochs
        self.intercept = intercept
        self.C = C
        self.tracker = tracker
        self.batch_size = batch_size
        self.threshold = threshold
        self.random_state = random_state
        self.w = None

    def __calculate_gradient(self, X, y):
        if self.penalty == "l2":
            regulator = self.w
        elif self.penalty == "l1":
            regulator = torch.sign(self.w)

        return (
            self.C * torch.mv(X.t(), self.__sigmoid(torch.mv(X, self.w)) - y)
            + regulator
        )

    def __sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def __add_intercept(self, X):
        return torch.cat([X.new_ones(X.size()[0], 1), X], dim=1)

    def __calculate_loss(self, X, y):
        if self.penalty == "l2":
            reg_loss = torch.dot(self.w, self.w).item()
        elif self.penalty == "l1":
            reg_loss = torch.sum(torch.abs(self.w)).item()

        a = self.__sigmoid(torch.mv(X, self.w))
        return (
            self.C * torch.mean(-(y * torch.log(a) + (1 - y) * torch.log(1 - a))).item()
            + reg_loss
        )

    @measure_time
    def fit(self, X, y):
        if self.intercept:
            X = self.__add_intercept(X)

        self.w = X.new_zeros(X.shape[1])
        self.loss = [self.__calculate_loss(X, y)]
        w_old = self.w.clone()
        w_dist = 1 ** 10
        n_iter = 0
        N = X.size()[0]
        torch.manual_seed(self.random_state)

        while (n_iter < self.epochs) and (w_dist > self.weight_dist):
            indices = torch.randint(0, N, (N,))
            X, y = X[indices], y[indices]

            for i in range(0, N, self.batch_size):
                X_train, y_train = (
                    X[i : i + self.batch_size],
                    y[i : i + self.batch_size],
                )
                self.w -= self.learning_rate * self.__calculate_gradient(
                    X_train, y_train
                )
                self.loss.append(self.__calculate_loss(X_train, y_train))

                w_dist = torch.norm(self.w - w_old).item()
                w_old = self.w.clone()
                n_iter += 1

        self.tracker.loss.append(self.loss)
        self.tracker.lr.append(self.learning_rate)
        self.tracker.batch.append(self.batch_size)
        return self

    def predict(self, X):
        if self.intercept:
            X = self.__add_intercept(X)

        if self.w is None:
            raise AttributeError("Not fitted yet")

        y = self.__sigmoid(torch.mv(X, self.w))
        zeros = X.new_zeros(X.size()[0])
        ones = X.new_ones(X.size()[0])

        return torch.where(y >= self.threshold, ones, zeros)

    def predict_proba(self, X):
        if self.intercept:
            X = self.__add_intercept(X)

        if self.w is None:
            raise AttributeError("Not fitted yet")

        positive = self.__sigmoid(torch.mv(X, self.w))
        negative = 1 - positive
        return torch.cat([negative[:, None], positive[:, None]], dim=1)
