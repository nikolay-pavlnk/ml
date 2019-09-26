import numpy as np
from logistic_regression import LogisticRegression
from utils import bin_feat_heart, con_feat_heart, name_features_heart
import utilities


TRAIN = 'heart_train.csv'
TEST = 'heart_test.csv'

if __name__ == '__main__':
	path = utilities.get_path()
	X_train, y_train = utilities.get_data(path/TRAIN, 10)
	X_test, y_test = utilities.get_data(path/TEST, 10)

	encoder = utilities.OneHotEncoder()
	scaler = utilities.StandardScaler()

	encoder.fit(X_train[:, bin_feat_heart])

	X_train_new = np.hstack((encoder.transform(X_train[:, bin_feat_heart]),
							X_train[:, con_feat_heart]))

	X_test_new = np.hstack((encoder.transform(X_test[:, bin_feat_heart]),
							X_test[:, con_feat_heart]))

	scaler.fit(X_train_new)
	X_train_scaled = scaler.transform(X_train_new)
	X_test_scaled = scaler.transform(X_test_new)

	model = LogisticRegression(C=3, learning_rate=0.0001)
	model.fit(X_train_scaled, y_train)

	print('Train metrics')
	utilities.classification_report(y_train, model.predict(X_train_scaled))
	print('Test metrics')
	utilities.classification_report(y_test, model.predict(X_test_scaled))

	print('Feature importances')

	args = np.argsort(np.fabs(model.w))[::-1]
	for i in args[:5]:
		print(name_features_heart[i], model.w[i])


