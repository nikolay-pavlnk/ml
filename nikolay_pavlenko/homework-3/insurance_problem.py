import numpy as np
from linear_regression import LinearRegression
from utils import mapper, bin_feat_reg, con_feat_reg, name_features_insurance
import utilities


TRAIN = 'insurance_train.csv'
TEST = 'insurance_test.csv'

if __name__ == '__main__':
	path = utilities.get_path()

	X_train, y_train = utilities.get_data(path/TRAIN, 2, mapper)
	X_test, y_test = utilities.get_data(path/TEST, 2, mapper)

	encoder = utilities.OneHotEncoder()
	scaler = utilities.StandardScaler()

	encoder.fit(X_train[:, bin_feat_reg])

	X_train_new = np.hstack((encoder.transform(X_train[:, bin_feat_reg]),
							X_train[:, con_feat_reg]))

	X_test_new = np.hstack((encoder.transform(X_test[:, bin_feat_reg]),
							X_test[:, con_feat_reg]))

	scaler.fit(X_train_new)
	X_train_scaled = scaler.transform(X_train_new)
	X_test_scaled = scaler.transform(X_test_new)

	model = LinearRegression(learning_rate=10e-5, penalty='l2')
	model.fit(X_train_scaled, y_train)

	print('Train metrics')
	utilities.regression_report(y_train, model.predict(X_train_scaled))
	print('Test metrics')
	utilities.regression_report(y_test, model.predict(X_test_scaled))

	print('Feature importances')

	args = np.argsort(np.fabs(model.w))[::-1]
	for i in args[:5]:
		print(name_features_insurance[i], model.w[i])
