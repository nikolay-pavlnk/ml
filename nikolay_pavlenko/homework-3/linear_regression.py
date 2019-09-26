import numpy as np
from metrics import MSE
from error import PredictError, NormalEquationError


class LinearRegression:
	def __init__(self, learning_rate=10e-3, weight_dist=10e-4, epochs=10e3, 
				intercept=True, normal_equation=False, alpha=1, penalty='l2', scorer=MSE):

		"""
		Parameters
		----------
		learning_rate: float [0, 1]
			step of gradient descent.
		weight_dist: float
			Distance between vector of parameters k + 1 and k.
		epochs: int
			Number of iteration for gradient descent.
		intercept: boolean
			True - Add intercept(fictitious column).
			False - Ignore.
		normal_equation: boolean
			True - Use an analytical solution.
			False - Use stochastic gradient descent.
		alpha_: float
			Regularization Coefficient for L2/L1 norm.
		penalty: str
			It indicates which regularization model have to use
		"""
		self.learning_rate = learning_rate
		self.weight_dist = weight_dist
		self.epochs = epochs
		self.scorer= scorer
		self.intercept = intercept
		self.normal_equation = normal_equation
		self.alpha = alpha
		self.penalty = penalty
		self.w = None

	def __calculate_gradient(self, X, y):
		"""
		Since we don't have a priori any knowledge about the interception, we don't regulate it.

		Parameters
		----------
		X: numpy array shape = (N, D+1)
			Matrix 'objects-features'. 
		Y: numpy array (shape = N, 1)
			Vector with labels.
		----------
		Return gradient of loss function
		"""
		if self.penalty is 'l2':
			regulator = np.hstack((self.w[0][np.newaxis], self.alpha * self.w[1:]))
		elif self.penalty is 'l1':
			regulator = np.hstack((self.w[0][np.newaxis], self.alpha * np.sign(self.w[1:])))

		return np.dot((np.dot(X, self.w) - y), X) + regulator

	def __add_intercept(self, X): 
		"""
		Add intercept to matrix X. 
		"""
		return np.hstack((np.ones((X.shape[0], 1)), X))

	def __get_analytical_solution(self, X, y):
		if self.penalty is 'l1':
			raise NormalEquationError('Solution with lasso regulator does not exist')

		if self.alpha_ridge == 0:
			print('WARNING: Ridge coefficient is equal to zero')

		self.w = np.dot(np.linalg.inv(np.dot(X.T, X) + self.alpha_ridge * np.identity(X.shape[1])), np.dot(X.T, y))
		return self

	def fit(self, X, y):
		"""
		Fit for training data

		Parameters
		----------
		X: numpy array shape = (N, D+1)
			Matrix 'objects-features'. 
		Y: numpy array (shape = N, 1)
			Vector with labels.
		----------
		"""
		if self.intercept:
			X = self.__add_intercept(X)

		if self.normal_equation:
			return self.__get_analytical_solution(X, y)

		self.w = np.zeros(X.shape[1])
		w_old = self.w.copy()
		w_dist = np.inf
		n_iter = 0

		while (n_iter < self.epochs) and (w_dist > self.weight_dist):

			self.w -= self.learning_rate * self.__calculate_gradient(X, y)
			w_dist = np.linalg.norm(self.w - w_old)

			w_old = self.w.copy()
			n_iter += 1

		return self

	def score(self, X, y_true):
		return self.scorer(y_true, self.predict(X))

	def predict(self, X):
		"""
		Predict the continuous number.
		"""
		if self.intercept:
			X = self.__add_intercept(X)

		if self.w is None:
			raise PredictError('Not fitted yet')

		return np.dot(X, self.w)
