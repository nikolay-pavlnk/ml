import numpy as np
from metrics import f1_score
from error import PredictError


class LogisticRegression:
	def __init__(self, C=1, penalty='l2', learning_rate=0.5, weight_dist=10e-4, epochs=10e5, intercept=True,
			scorer=f1_score, threshold=0.5):
		"""
		Parameters
		----------
		C: float
			Regularization Coefficient. C = 1 \ alpha.
		penalty: str ('l1', 'l2')
			The parameter shows which regularization norms should be used to fit the model.
		learning_rate: float [0, 1]
			step of gradient descent.
		weight_dist: float
			Distance between vector of parameters k + 1 and k. It used for stop criterion.
		epochs: int
			Number of iteration for gradient descent.
		intercept: boolean
			True - Add intercept(fictitious column).
			False - Ignore.
		scorer: function
			It indicates which metric should be optimized in GridSearch.
		threshold: float:
			The threshold at which the objects with  predicted probabilities splitted into labels.
		"""
		self.learning_rate = learning_rate
		self.scorer = scorer
		self.penalty = penalty
		self.weight_dist = weight_dist
		self.epochs = epochs
		self.intercept = intercept
		self.C = C
		self.threshold = threshold
		self.w = None

	def __calculate_gradient(self, X, y):
		"""
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
			regulator = self.w
		elif self.penalty is 'l1':
			regulator = np.sign(self.w)

		return (self.C * np.dot(X.T, (self.__sigmoid(np.dot(X, self.w)) - y))) + regulator

	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def score(self, X, y_true):
		return self.scorer(y_true, self.predict(X))

	def __add_intercept(self, X):
		"""
		Add intercept to matrix X. 
		"""
		return np.hstack((np.ones((X.shape[0], 1)), X))

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

	def predict(self, X):
		"""
		Predict labels of the objects
		"""
		if self.intercept:
			X = self.__add_intercept(X)

		if self.w is None:
			raise PredictError('Not fitted yet')

		y = self.__sigmoid(np.dot(X, self.w))
		return np.where(y >= self.threshold, 1, 0)

	def predict_proba(self, X):
		"""
		Predict the probability that the object belongs to class 1.
		"""
		if self.intercept:
			X = self.__add_intercept(X)

		if self.w is None:
			raise PredictError('Not fitted yet')

		positive = self.__sigmoid(np.dot(X, self.w))
		negative = 1 - positive
		return np.hstack((negative[:, np.newaxis], positive[:, np.newaxis]))
